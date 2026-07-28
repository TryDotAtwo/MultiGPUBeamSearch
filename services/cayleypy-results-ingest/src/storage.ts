import { canonicalJson, computeIdempotency, newSubmissionId, rawObjectKey, sha256Hex } from "./ids.js";
import { findByIdempotency, findBySubmissionId, transition, type SubmissionRow, type SubmissionState } from "./db.js";
import type { ResultEnvelopeV1 } from "./schema.js";

export interface IngestEnv { RESULTS_DB: D1Database; RAW_RESULTS: R2Bucket; VALIDATE_QUEUE: Queue }
export interface RequestMeta { receivedAt?: Date; duplicatePollAttempts?: number; duplicatePollDelayMs?: number }
export interface Receipt { submission_id: string; idempotency_key: string; state: "queued" | "retryable"; duplicate: boolean }

export type SafeIngestErrorCode =
  | "raw_object_conflict"
  | "submission_lookup_failed"
  | "submission_persist_failed"
  | "duplicate_resolution_failed"
  | "duplicate_wait_timeout"
  | "duplicate_raw_cleanup_failed"
  | "state_transition_failed"
  | "state_transition_conflict";

export class SafeIngestError extends Error {
  constructor(readonly code: SafeIngestErrorCode) { super(code); }
}

const QUEUED_OR_LATER = new Set<SubmissionState>([
  "queued", "validating", "validated", "rejected", "staged", "published", "dead_letter",
]);
const DEFAULT_DUPLICATE_POLL_ATTEMPTS = 100;
const DEFAULT_DUPLICATE_POLL_DELAY_MS = 10;
const MAX_DUPLICATE_POLL_ATTEMPTS = 200;
const MAX_DUPLICATE_POLL_DELAY_MS = 100;

async function putRaw(bucket: R2Bucket, key: string, body: string, sha256: string): Promise<void> {
  const onlyIf = new Headers({ "If-None-Match": "*" });
  const written = await bucket.put(key, body, { onlyIf, httpMetadata: { contentType: "application/json; charset=utf-8" }, customMetadata: { sha256 } });
  if (written !== null) return;
  throw new SafeIngestError("raw_object_conflict");
}

function boundedInteger(value: number | undefined, fallback: number, maximum: number): number {
  if (value === undefined || !Number.isInteger(value)) return fallback;
  return Math.max(0, Math.min(value, maximum));
}

function settledReceipt(row: SubmissionRow, duplicate: boolean): Receipt | null {
  if (row.state === "retryable") {
    return { submission_id: row.submission_id, idempotency_key: row.idempotency_key, state: "retryable", duplicate };
  }
  if (QUEUED_OR_LATER.has(row.state)) {
    return { submission_id: row.submission_id, idempotency_key: row.idempotency_key, state: "queued", duplicate };
  }
  return null;
}

async function delay(milliseconds: number): Promise<void> {
  if (milliseconds === 0) { await Promise.resolve(); return; }
  await new Promise<void>((resolve) => setTimeout(resolve, milliseconds));
}

async function waitForDuplicate(env: IngestEnv, key: string, first: SubmissionRow, requestMeta: RequestMeta): Promise<Receipt> {
  const attempts = Math.max(1, boundedInteger(requestMeta.duplicatePollAttempts, DEFAULT_DUPLICATE_POLL_ATTEMPTS, MAX_DUPLICATE_POLL_ATTEMPTS));
  const delayMs = boundedInteger(requestMeta.duplicatePollDelayMs, DEFAULT_DUPLICATE_POLL_DELAY_MS, MAX_DUPLICATE_POLL_DELAY_MS);
  let row = first;
  for (let attempt = 0; attempt <= attempts; attempt += 1) {
    const receipt = settledReceipt(row, true);
    if (receipt) return receipt;
    if (row.state !== "received") throw new SafeIngestError("state_transition_conflict");
    if (attempt === attempts) break;
    await delay(delayMs);
    try {
      const reread = await findByIdempotency(env.RESULTS_DB, key);
      if (!reread) throw new SafeIngestError("duplicate_resolution_failed");
      row = reread;
    } catch (error) {
      if (error instanceof SafeIngestError) throw error;
      throw new SafeIngestError("submission_lookup_failed");
    }
  }
  throw new SafeIngestError("duplicate_wait_timeout");
}

async function cleanupDuplicateRaw(bucket: R2Bucket, key: string): Promise<void> {
  try {
    await bucket.delete(key);
    if (await bucket.head(key) !== null) throw new Error("cleanup_unconfirmed");
  } catch {
    throw new SafeIngestError("duplicate_raw_cleanup_failed");
  }
}

async function safeReadSubmission(db: D1Database, id: string): Promise<SubmissionRow> {
  try {
    const row = await findBySubmissionId(db, id);
    if (!row) throw new SafeIngestError("state_transition_conflict");
    return row;
  } catch (error) {
    if (error instanceof SafeIngestError) throw error;
    throw new SafeIngestError("state_transition_failed");
  }
}

async function markAfterQueueSuccess(env: IngestEnv, submissionId: string, idempotencyKey: string): Promise<Receipt> {
  let transitioned: boolean;
  try {
    transitioned = await transition(env.RESULTS_DB, submissionId, ["received"], "queued");
  } catch {
    throw new SafeIngestError("state_transition_failed");
  }
  if (transitioned) return { submission_id: submissionId, idempotency_key: idempotencyKey, state: "queued", duplicate: false };
  const row = await safeReadSubmission(env.RESULTS_DB, submissionId);
  if (QUEUED_OR_LATER.has(row.state)) {
    return { submission_id: submissionId, idempotency_key: idempotencyKey, state: "queued", duplicate: false };
  }
  throw new SafeIngestError("state_transition_conflict");
}

async function markAfterQueueFailure(env: IngestEnv, submissionId: string, idempotencyKey: string): Promise<Receipt> {
  let transitioned: boolean;
  try {
    transitioned = await transition(env.RESULTS_DB, submissionId, ["received"], "retryable", { safeError: "queue_unavailable" });
  } catch {
    throw new SafeIngestError("state_transition_failed");
  }
  if (transitioned) return { submission_id: submissionId, idempotency_key: idempotencyKey, state: "retryable", duplicate: false };
  const row = await safeReadSubmission(env.RESULTS_DB, submissionId);
  if (row.state === "retryable") {
    return { submission_id: submissionId, idempotency_key: idempotencyKey, state: "retryable", duplicate: false };
  }
  throw new SafeIngestError("state_transition_conflict");
}

/** Durably stores raw JSON before D1 received, awaits Queue durability, then confirms the receipt state. */
export async function receiveEnvelope(env: IngestEnv, envelope: ResultEnvelopeV1, requestMeta: RequestMeta = {}): Promise<Receipt> {
  const idempotencyKey = await computeIdempotency(envelope);
  let prior: SubmissionRow | null;
  try {
    prior = await findByIdempotency(env.RESULTS_DB, idempotencyKey);
  } catch {
    throw new SafeIngestError("submission_lookup_failed");
  }
  if (prior) return waitForDuplicate(env, idempotencyKey, prior, requestMeta);

  const now = requestMeta.receivedAt ?? new Date();
  const submissionId = newSubmissionId(now);
  const key = rawObjectKey(submissionId, now);
  const rawBody = canonicalJson(envelope);
  const rawSha256 = await sha256Hex(rawBody);
  await putRaw(env.RAW_RESULTS, key, rawBody, rawSha256);

  const timestamp = now.toISOString();
  let inserted: boolean;
  try {
    const insert = await env.RESULTS_DB.prepare(
      "INSERT INTO submissions (submission_id,idempotency_key,run_id,author_name,competition,puzzle_type,puzzle_id,state,raw_r2_key,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(idempotency_key) DO NOTHING",
    ).bind(submissionId, idempotencyKey, envelope.run_id, envelope.author.name, envelope.competition, envelope.puzzle_type, envelope.puzzle_id, "received", key, timestamp, timestamp).run();
    inserted = insert.meta.changes === 1;
  } catch {
    throw new SafeIngestError("submission_persist_failed");
  }
  if (!inserted) {
    let winner: SubmissionRow | null;
    try {
      winner = await findByIdempotency(env.RESULTS_DB, idempotencyKey);
    } catch {
      await cleanupDuplicateRaw(env.RAW_RESULTS, key);
      throw new SafeIngestError("submission_lookup_failed");
    }
    await cleanupDuplicateRaw(env.RAW_RESULTS, key);
    if (!winner) throw new SafeIngestError("duplicate_resolution_failed");
    return waitForDuplicate(env, idempotencyKey, winner, requestMeta);
  }

  let queueFailed = false;
  try {
    await env.VALIDATE_QUEUE.send({ submission_id: submissionId });
  } catch {
    queueFailed = true;
  }
  if (queueFailed) return markAfterQueueFailure(env, submissionId, idempotencyKey);
  return markAfterQueueSuccess(env, submissionId, idempotencyKey);
}
