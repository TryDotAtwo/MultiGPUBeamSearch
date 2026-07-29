import {
  findByIdempotency,
  findBySubmissionId,
  findStaleRecoverable,
  transition,
  type SubmissionRow,
  type SubmissionState,
} from "./db.js";
import { canonicalJson, computeIdempotency, newSubmissionId, rawObjectKey, sha256Hex } from "./ids.js";
import { validateEnvelopeIntegrity, type ResultEnvelopeV1 } from "./schema.js";

export interface IngestEnv {
  RESULTS_DB: D1Database;
  RAW_RESULTS: R2Bucket;
  VALIDATE_QUEUE: Queue;
}

export interface PersistenceEnv {
  RESULTS_DB: D1Database;
  RAW_RESULTS: R2Bucket;
}

export interface RequestMeta {
  receivedAt?: Date;
  duplicatePollAttempts?: number;
  duplicatePollDelayMs?: number;
}

export interface Receipt {
  submission_id: string;
  idempotency_key: string;
  state: "queued" | "retryable";
  duplicate: boolean;
}

export interface StoredReceipt {
  submission_id: string;
  idempotency_key: string;
  state: "received" | "queued" | "retryable";
  duplicate: boolean;
}

export interface RecoveryOptions {
  staleBefore: Date;
  limit?: number;
}

export interface RecoverySummary {
  scanned: number;
  queued: number;
  retryable: number;
  failed: number;
}

export type SafeIngestErrorCode =
  | "raw_object_conflict"
  | "submission_lookup_failed"
  | "submission_persist_failed"
  | "duplicate_resolution_failed"
  | "duplicate_raw_cleanup_failed"
  | "state_transition_failed"
  | "state_transition_conflict"
  | "recovery_query_failed"
  | "invalid_envelope";

export class SafeIngestError extends Error {
  constructor(readonly code: SafeIngestErrorCode) {
    super(code);
  }
}

const QUEUED_OR_LATER = new Set<SubmissionState>([
  "queued",
  "validating",
  "validated",
  "rejected",
  "staged",
  "published",
  "dead_letter",
]);
const DEFAULT_DUPLICATE_POLL_ATTEMPTS = 100;
const DEFAULT_DUPLICATE_POLL_DELAY_MS = 10;
const MAX_DUPLICATE_POLL_ATTEMPTS = 200;
const MAX_DUPLICATE_POLL_DELAY_MS = 100;
const DEFAULT_RECOVERY_LIMIT = 50;
const MAX_RECOVERY_LIMIT = 100;

async function putRaw(bucket: R2Bucket, key: string, body: string, sha256: string): Promise<void> {
  const written = await bucket.put(key, body, {
    onlyIf: new Headers({ "If-None-Match": "*" }),
    httpMetadata: { contentType: "application/json; charset=utf-8" },
    customMetadata: { sha256 },
  });
  if (written === null) throw new SafeIngestError("raw_object_conflict");
}

function boundedInteger(value: number | undefined, fallback: number, maximum: number): number {
  if (value === undefined || !Number.isInteger(value)) return fallback;
  return Math.max(0, Math.min(value, maximum));
}

function settledReceipt(row: SubmissionRow, duplicate: boolean): Receipt | null {
  if (row.state === "retryable") {
    return {
      submission_id: row.submission_id,
      idempotency_key: row.idempotency_key,
      state: "retryable",
      duplicate,
    };
  }
  if (QUEUED_OR_LATER.has(row.state)) {
    return {
      submission_id: row.submission_id,
      idempotency_key: row.idempotency_key,
      state: "queued",
      duplicate,
    };
  }
  return null;
}

async function delay(ms: number): Promise<void> {
  if (ms === 0) {
    await Promise.resolve();
    return;
  }
  await new Promise<void>((resolve) => setTimeout(resolve, ms));
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

export async function parkPausedSubmission(
  db: D1Database,
  submissionId: string,
): Promise<SubmissionRow> {
  let changed: boolean;
  try {
    changed = await transition(
      db,
      submissionId,
      ["received", "queued", "retryable"],
      "retryable",
      { safeError: "ingest_paused" },
    );
  } catch {
    throw new SafeIngestError("state_transition_failed");
  }

  const current = await safeReadSubmission(db, submissionId);
  if (current.state === "retryable" && current.safe_error === "ingest_paused") {
    return current;
  }
  throw new SafeIngestError(changed ? "state_transition_failed" : "state_transition_conflict");
}

async function confirmQueueSuccess(env: IngestEnv, row: SubmissionRow, duplicate: boolean): Promise<Receipt> {
  let changed: boolean;
  try {
    changed = await transition(
      env.RESULTS_DB,
      row.submission_id,
      ["received", "queued", "retryable"],
      "queued",
      { safeError: null },
    );
  } catch {
    throw new SafeIngestError("state_transition_failed");
  }
  if (changed) {
    return {
      submission_id: row.submission_id,
      idempotency_key: row.idempotency_key,
      state: "queued",
      duplicate,
    };
  }

  const current = await safeReadSubmission(env.RESULTS_DB, row.submission_id);
  if (QUEUED_OR_LATER.has(current.state)) {
    return {
      submission_id: current.submission_id,
      idempotency_key: current.idempotency_key,
      state: "queued",
      duplicate,
    };
  }
  throw new SafeIngestError("state_transition_conflict");
}

async function confirmQueueFailure(env: IngestEnv, row: SubmissionRow, duplicate: boolean): Promise<Receipt> {
  let current = await safeReadSubmission(env.RESULTS_DB, row.submission_id);
  if (current.state !== "queued" && QUEUED_OR_LATER.has(current.state)) {
    return {
      submission_id: current.submission_id,
      idempotency_key: current.idempotency_key,
      state: "queued",
      duplicate,
    };
  }
  if (current.state !== "received" && current.state !== "queued" && current.state !== "retryable") {
    throw new SafeIngestError("state_transition_conflict");
  }

  let changed: boolean;
  try {
    changed = await transition(
      env.RESULTS_DB,
      row.submission_id,
      ["received", "queued", "retryable"],
      "retryable",
      { safeError: "queue_unavailable", incrementRetryCount: true },
    );
  } catch {
    throw new SafeIngestError("state_transition_failed");
  }
  if (changed) {
    return {
      submission_id: row.submission_id,
      idempotency_key: row.idempotency_key,
      state: "retryable",
      duplicate,
    };
  }

  current = await safeReadSubmission(env.RESULTS_DB, row.submission_id);
  if (current.state !== "queued" && QUEUED_OR_LATER.has(current.state)) {
    return {
      submission_id: current.submission_id,
      idempotency_key: current.idempotency_key,
      state: "queued",
      duplicate,
    };
  }
  if (current.state === "retryable") {
    return {
      submission_id: current.submission_id,
      idempotency_key: current.idempotency_key,
      state: "retryable",
      duplicate,
    };
  }
  throw new SafeIngestError("state_transition_conflict");
}

async function resendExisting(env: IngestEnv, row: SubmissionRow, duplicate: boolean): Promise<Receipt> {
  try {
    await env.VALIDATE_QUEUE.send({ submission_id: row.submission_id });
  } catch {
    return confirmQueueFailure(env, row, duplicate);
  }
  return confirmQueueSuccess(env, row, duplicate);
}

async function waitForDuplicate(
  env: IngestEnv,
  key: string,
  first: SubmissionRow,
  meta: RequestMeta,
): Promise<Receipt> {
  const attempts = Math.max(
    1,
    boundedInteger(meta.duplicatePollAttempts, DEFAULT_DUPLICATE_POLL_ATTEMPTS, MAX_DUPLICATE_POLL_ATTEMPTS),
  );
  const delayMs = boundedInteger(
    meta.duplicatePollDelayMs,
    DEFAULT_DUPLICATE_POLL_DELAY_MS,
    MAX_DUPLICATE_POLL_DELAY_MS,
  );
  let row = first;

  for (let attempt = 0; attempt <= attempts; attempt += 1) {
    const receipt = settledReceipt(row, true);
    if (receipt) return receipt;
    if (row.state !== "received") throw new SafeIngestError("state_transition_conflict");
    if (attempt === attempts) return resendExisting(env, row, true);

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

  throw new SafeIngestError("state_transition_conflict");
}

async function cleanupDuplicateRaw(bucket: R2Bucket, key: string): Promise<void> {
  try {
    await bucket.delete(key);
    if (await bucket.head(key) !== null) throw new Error("cleanup_unconfirmed");
  } catch {
    throw new SafeIngestError("duplicate_raw_cleanup_failed");
  }
}

export async function recoverStaleSubmissions(
  env: IngestEnv,
  options: RecoveryOptions,
): Promise<RecoverySummary> {
  const limit = Math.max(1, boundedInteger(options.limit, DEFAULT_RECOVERY_LIMIT, MAX_RECOVERY_LIMIT));
  let rows: SubmissionRow[];
  try {
    rows = await findStaleRecoverable(env.RESULTS_DB, options.staleBefore.toISOString(), limit);
  } catch {
    throw new SafeIngestError("recovery_query_failed");
  }

  const summary: RecoverySummary = { scanned: rows.length, queued: 0, retryable: 0, failed: 0 };
  for (const row of rows) {
    try {
      const receipt = await resendExisting(env, row, true);
      summary[receipt.state] += 1;
    } catch {
      summary.failed += 1;
    }
  }
  return summary;
}

export async function receiveEnvelope(
  env: IngestEnv,
  envelope: ResultEnvelopeV1,
  meta: RequestMeta = {},
): Promise<Receipt> {
  if ((await validateEnvelopeIntegrity(envelope)).length !== 0) throw new SafeIngestError("invalid_envelope");
  const idempotencyKey = await computeIdempotency(envelope);
  let prior: SubmissionRow | null;
  try {
    prior = await findByIdempotency(env.RESULTS_DB, idempotencyKey);
  } catch {
    throw new SafeIngestError("submission_lookup_failed");
  }
  if (prior) return waitForDuplicate(env, idempotencyKey, prior, meta);

  const now = meta.receivedAt ?? new Date();
  const submissionId = newSubmissionId(now);
  const key = rawObjectKey(submissionId, now);
  const rawBody = canonicalJson(envelope);
  await putRaw(env.RAW_RESULTS, key, rawBody, await sha256Hex(rawBody));

  const timestamp = now.toISOString();
  let inserted: boolean;
  try {
    const result = await env.RESULTS_DB.prepare(
      "INSERT INTO submissions (submission_id,idempotency_key,run_id,author_name,competition,puzzle_type,puzzle_id,state,raw_r2_key,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(idempotency_key) DO NOTHING",
    )
      .bind(
        submissionId,
        idempotencyKey,
        envelope.run_id,
        envelope.author.name,
        envelope.competition,
        envelope.puzzle_type,
        envelope.puzzle_id,
        "received",
        key,
        timestamp,
        timestamp,
      )
      .run();
    inserted = result.meta.changes === 1;
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
    return waitForDuplicate(env, idempotencyKey, winner, meta);
  }

  return resendExisting(
    env,
    {
      submission_id: submissionId,
      idempotency_key: idempotencyKey,
      state: "received",
      raw_r2_key: key,
      safe_error: null,
      retry_count: 0,
      updated_at: timestamp,
    },
    false,
  );
}

function storedReceipt(row: SubmissionRow, duplicate: boolean): StoredReceipt {
  if (row.state === "received" || row.state === "retryable") {
    return {
      submission_id: row.submission_id,
      idempotency_key: row.idempotency_key,
      state: row.state,
      duplicate,
    };
  }
  if (QUEUED_OR_LATER.has(row.state)) {
    return {
      submission_id: row.submission_id,
      idempotency_key: row.idempotency_key,
      state: "queued",
      duplicate,
    };
  }
  throw new SafeIngestError("state_transition_conflict");
}

/** Persist an accepted envelope without touching Queue publication. */
export async function receiveEnvelopeStoreOnly(
  env: PersistenceEnv,
  envelope: ResultEnvelopeV1,
  meta: Pick<RequestMeta, "receivedAt"> = {},
): Promise<StoredReceipt> {
  if ((await validateEnvelopeIntegrity(envelope)).length !== 0) throw new SafeIngestError("invalid_envelope");
  const idempotencyKey = await computeIdempotency(envelope);
  let prior: SubmissionRow | null;
  try {
    prior = await findByIdempotency(env.RESULTS_DB, idempotencyKey);
  } catch {
    throw new SafeIngestError("submission_lookup_failed");
  }
  if (prior) return storedReceipt(prior, true);

  const now = meta.receivedAt ?? new Date();
  const submissionId = newSubmissionId(now);
  const key = rawObjectKey(submissionId, now);
  const rawBody = canonicalJson(envelope);
  await putRaw(env.RAW_RESULTS, key, rawBody, await sha256Hex(rawBody));

  const timestamp = now.toISOString();
  let inserted: boolean;
  try {
    const result = await env.RESULTS_DB.prepare(
      "INSERT INTO submissions (submission_id,idempotency_key,run_id,author_name,competition,puzzle_type,puzzle_id,state,raw_r2_key,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(idempotency_key) DO NOTHING",
    )
      .bind(
        submissionId,
        idempotencyKey,
        envelope.run_id,
        envelope.author.name,
        envelope.competition,
        envelope.puzzle_type,
        envelope.puzzle_id,
        "received",
        key,
        timestamp,
        timestamp,
      )
      .run();
    inserted = result.meta.changes === 1;
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
    return storedReceipt(winner, true);
  }

  return {
    submission_id: submissionId,
    idempotency_key: idempotencyKey,
    state: "received",
    duplicate: false,
  };
}
