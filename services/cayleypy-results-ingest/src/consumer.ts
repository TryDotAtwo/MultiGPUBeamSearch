import { findBySubmissionId, transition, type SubmissionRow } from "./db.js";
import { canonicalJson, sha256Hex } from "./ids.js";
import { MAX_SERIALIZED_ENVELOPE_BYTES } from "./schema.js";
import { validateVersionedBatch, validateVersionedEnvelope, type ResultEnvelope, type SchemaVersion } from "./schema-dispatch.js";
import { parkPausedSubmission, SafeIngestError, type IngestEnv } from "./storage.js";

export type ConsumerMode = "normal" | "store_only" | "reject";
// Cloudflare attempts includes initial delivery; wrangler max_retries=8 means 9 attempts.
export const MAX_QUEUE_ATTEMPTS = 9;
export const MAX_RETRY_DELAY_SECONDS = 300;

export interface QueueMessageLike {
  body: unknown;
  attempts?: number;
  ack(): void;
  retry(options?: { delaySeconds?: number }): void;
}

export interface GitHubWriterNamespace {
  getByName(name: string): { enqueueValidated(submissionId: string): Promise<void> };
}

export interface ConsumerEnv extends IngestEnv {
  INGEST_MODE?: string;
  VALIDATE_DLQ?: Queue;
  GITHUB_WRITER?: GitHubWriterNamespace;
}

export function queueSubmissionId(body: unknown): string | null {
  if (body === null || typeof body !== "object" || Array.isArray(body)) return null;
  const id = (body as Record<string, unknown>).submission_id;
  return typeof id === "string" && /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(id) ? id : null;
}

export function retryDelaySeconds(retryCount: number): number {
  const exponent = Math.max(0, Math.min(8, Math.trunc(retryCount)));
  return Math.min(MAX_RETRY_DELAY_SECONDS, 2 ** exponent);
}

function isFinalRepositoryState(state: SubmissionRow["state"]): boolean {
  return state === "rejected" || state === "staged" || state === "published";
}

function exhausted(message: QueueMessageLike): boolean {
  return (message.attempts ?? 1) >= MAX_QUEUE_ATTEMPTS;
}

async function enqueueValidated(env: ConsumerEnv, submissionId: string): Promise<void> {
  if (env.GITHUB_WRITER === undefined) throw new Error("github_writer_unavailable");
  await env.GITHUB_WRITER.getByName("cayleypy-results-v1").enqueueValidated(submissionId);
}

async function markRetryable(env: ConsumerEnv, id: string, code: string): Promise<number> {
  const changed = await transition(env.RESULTS_DB, id, ["received", "queued", "validating", "retryable"], "retryable", {
    safeError: code, incrementRetryCount: true,
  });
  if (!changed) {
    const current = await findBySubmissionId(env.RESULTS_DB, id);
    if (!current || current.state === "validated" || current.state === "dead_letter" || isFinalRepositoryState(current.state)) {
      return current?.retry_count ?? 0;
    }
    throw new SafeIngestError("state_transition_conflict");
  }
  const current = await findBySubmissionId(env.RESULTS_DB, id);
  if (!current) throw new SafeIngestError("state_transition_conflict");
  return current.retry_count;
}

async function deliverDeadLetter(message: QueueMessageLike, env: ConsumerEnv, submissionId: string): Promise<void> {
  try {
    if (env.VALIDATE_DLQ === undefined) throw new Error("validate_dlq_unavailable");
    await env.VALIDATE_DLQ.send({ submission_id: submissionId });
    // The durable transition already happened before the side effect. Do not
    // clear safe_error here: it is the forensic reason for this DLQ item.
    message.ack();
  } catch {
    // Explicit DLQ delivery failed; platform retry/DLQ is the lossless fallback.
    message.retry();
  }
}

async function retryOrDeadLetter(message: QueueMessageLike, env: ConsumerEnv, submissionId: string, code: string): Promise<void> {
  if (exhausted(message)) {
    try {
      const changed = await transition(env.RESULTS_DB, submissionId, ["received", "queued", "validating", "retryable", "validated"], "dead_letter", {
        safeError: code, incrementRetryCount: true,
      });
      if (!changed) {
        const current = await findBySubmissionId(env.RESULTS_DB, submissionId);
        if (!current || current.state !== "dead_letter") throw new SafeIngestError("state_transition_conflict");
      }
      await deliverDeadLetter(message, env, submissionId);
    } catch { message.retry(); }
    return;
  }
  try {
    const retryCount = await markRetryable(env, submissionId, code);
    message.retry({ delaySeconds: retryDelaySeconds(retryCount) });
  } catch { message.retry({ delaySeconds: retryDelaySeconds(0) }); }
}

async function reject(env: ConsumerEnv, id: string, code: string): Promise<void> {
  const changed = await transition(env.RESULTS_DB, id, ["received", "queued", "validating", "retryable"], "rejected", { safeError: code });
  if (!changed) {
    const current = await findBySubmissionId(env.RESULTS_DB, id);
    if (!current || !isFinalRepositoryState(current.state)) throw new SafeIngestError("state_transition_conflict");
  }
}

async function validateRawEnvelope(env: ConsumerEnv, row: SubmissionRow): Promise<"valid" | "invalid"> {
  const object = await env.RAW_RESULTS.get(row.raw_r2_key);
  if (object === null) throw new Error("raw_missing");
  let candidate: unknown;
  if (object.size > MAX_SERIALIZED_ENVELOPE_BYTES) return "invalid";
  // R2's immutable object digest is the authoritative transport integrity check.
  // Read exactly once so the verified bytes are also the parsed bytes.
  const raw = await object.text();
  const storedHash = object.customMetadata?.sha256;
  if (new TextEncoder().encode(raw).byteLength > MAX_SERIALIZED_ENVELOPE_BYTES) return "invalid";
  if (typeof storedHash !== "string" || !/^[0-9a-f]{64}$/.test(storedHash) || await sha256Hex(raw) !== storedHash) return "invalid";
  try { candidate = JSON.parse(raw); } catch { return "invalid"; }
  let encoded: number;
  try { encoded = new TextEncoder().encode(canonicalJson({ schema_version: 1, results: [candidate] })).byteLength; } catch { return "invalid"; }
  const version: SchemaVersion = candidate !== null && typeof candidate === "object" && (candidate as Record<string, unknown>).schema_version === 2 ? 2 : 1;
  const batch = validateVersionedBatch({ schema_version: version, results: [candidate] }, version, encoded);
  if (!batch.ok) return "invalid";
  const envelope = batch.value.results[0] as ResultEnvelope;
  if (envelope.idempotency_key !== row.idempotency_key) return "invalid";
  return (await validateVersionedEnvelope(envelope)).length === 0 ? "valid" : "invalid";
}

async function enqueueValidatedOrRetry(message: QueueMessageLike, env: ConsumerEnv, submissionId: string): Promise<void> {
  try {
    await enqueueValidated(env, submissionId);
    const refreshed = await transition(env.RESULTS_DB, submissionId, ["validated"], "validated", { safeError: null });
    if (!refreshed) {
      const current = await findBySubmissionId(env.RESULTS_DB, submissionId);
      if (!current || !isFinalRepositoryState(current.state)) throw new SafeIngestError("state_transition_conflict");
    }
    message.ack();
  } catch {
    // Keep the row validated for all non-exhausted retries; never regress it.
    if (exhausted(message)) {
      await retryOrDeadLetter(message, env, submissionId, "publisher_unavailable");
      return;
    }
    try {
      const changed = await transition(env.RESULTS_DB, submissionId, ["validated"], "validated", {
        safeError: "publisher_unavailable", incrementRetryCount: true,
      });
      if (!changed) throw new SafeIngestError("state_transition_conflict");
      const current = await findBySubmissionId(env.RESULTS_DB, submissionId);
      message.retry({ delaySeconds: retryDelaySeconds(current?.retry_count ?? 0) });
    } catch { message.retry({ delaySeconds: retryDelaySeconds(0) }); }
  }
}

/** Queue handler for at-least-once validation. Non-normal modes never read R2. */
export async function consumeValidationMessage(message: QueueMessageLike, env: ConsumerEnv, mode: ConsumerMode): Promise<void> {
  if (mode !== "normal") {
    const submissionId = queueSubmissionId(message.body);
    if (submissionId === null) { message.ack(); return; }
    try {
      const current = await findBySubmissionId(env.RESULTS_DB, submissionId);
      if (current === null || current.state === "validating" || current.state === "validated" || current.state === "dead_letter" || isFinalRepositoryState(current.state)) {
        message.ack();
        return;
      }
      await parkPausedSubmission(env.RESULTS_DB, submissionId);
      message.ack();
    } catch { message.retry({ delaySeconds: retryDelaySeconds(0) }); }
    return;
  }

  const submissionId = queueSubmissionId(message.body);
  if (submissionId === null) { message.ack(); return; }
  let row: SubmissionRow | null;
  try { row = await findBySubmissionId(env.RESULTS_DB, submissionId); } catch { message.retry({ delaySeconds: retryDelaySeconds(0) }); return; }
  if (row === null || isFinalRepositoryState(row.state)) { message.ack(); return; }
  if (row.state === "dead_letter") { await deliverDeadLetter(message, env, submissionId); return; }
  if (row.state === "validated") { await enqueueValidatedOrRetry(message, env, submissionId); return; }

  try {
    const changed = await transition(env.RESULTS_DB, submissionId, ["received", "queued", "retryable"], "validating", { safeError: null });
    if (!changed) {
      row = await findBySubmissionId(env.RESULTS_DB, submissionId);
      if (row === null || isFinalRepositoryState(row.state)) { message.ack(); return; }
      if (row.state === "dead_letter") { await deliverDeadLetter(message, env, submissionId); return; }
      if (row.state === "validated") { await enqueueValidatedOrRetry(message, env, submissionId); return; }
      // A concurrent claimant owns this validating lease; duplicate delivery ACKs.
      if (row.state === "validating") { message.ack(); return; }
      throw new SafeIngestError("state_transition_conflict");
    }
    const current = row ?? await findBySubmissionId(env.RESULTS_DB, submissionId);
    if (current === null) throw new SafeIngestError("state_transition_conflict");
    if (await validateRawEnvelope(env, current) === "invalid") {
      await reject(env, submissionId, "invalid_envelope");
      message.ack();
      return;
    }
    const validated = await transition(env.RESULTS_DB, submissionId, ["validating"], "validated", { safeError: null });
    if (!validated) throw new SafeIngestError("state_transition_conflict");
    await enqueueValidatedOrRetry(message, env, submissionId);
  } catch {
    await retryOrDeadLetter(message, env, submissionId, "validation_unavailable");
  }
}
