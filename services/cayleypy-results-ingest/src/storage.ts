import { canonicalJson, computeIdempotency, newSubmissionId, rawObjectKey, sha256Hex } from "./ids.js";
import { findByIdempotency, transition, type SubmissionState } from "./db.js";
import type { ResultEnvelopeV1 } from "./schema.js";

export interface IngestEnv { RESULTS_DB: D1Database; RAW_RESULTS: R2Bucket; VALIDATE_QUEUE: Queue }
export interface RequestMeta { receivedAt?: Date }
export interface Receipt { submission_id: string; idempotency_key: string; state: "queued" | "retryable"; duplicate: boolean }

export class SafeIngestError extends Error { constructor(readonly code: "raw_object_conflict") { super(code); } }

async function putRaw(bucket: R2Bucket, key: string, body: string, sha256: string): Promise<void> {
  const onlyIf = new Headers({ "If-None-Match": "*" });
  const written = await bucket.put(key, body, { onlyIf, httpMetadata: { contentType: "application/json; charset=utf-8" }, customMetadata: { sha256 } });
  if (written !== null) return;
  const existing = await bucket.head(key);
  if (existing?.customMetadata?.sha256 === sha256) return;
  throw new SafeIngestError("raw_object_conflict");
}

function asReceipt(row: { submission_id: string; idempotency_key: string; state: SubmissionState }): Receipt {
  return { submission_id: row.submission_id, idempotency_key: row.idempotency_key, state: row.state === "retryable" ? "retryable" : "queued", duplicate: true };
}

/** Durably stores raw JSON before any D1 receipt and awaits Queue durability. */
export async function receiveEnvelope(env: IngestEnv, envelope: ResultEnvelopeV1, requestMeta: RequestMeta = {}): Promise<Receipt> {
  const idempotencyKey = await computeIdempotency(envelope);
  const prior = await findByIdempotency(env.RESULTS_DB, idempotencyKey);
  if (prior) return asReceipt(prior);

  const now = requestMeta.receivedAt ?? new Date();
  const submissionId = newSubmissionId(now);
  const key = rawObjectKey(submissionId, now);
  const rawBody = canonicalJson(envelope);
  const rawSha256 = await sha256Hex(rawBody);
  await putRaw(env.RAW_RESULTS, key, rawBody, rawSha256);

  const timestamp = now.toISOString();
  const insert = await env.RESULTS_DB.prepare(
    "INSERT INTO submissions (submission_id,idempotency_key,run_id,author_name,competition,puzzle_type,puzzle_id,state,raw_r2_key,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(idempotency_key) DO NOTHING",
  ).bind(submissionId, idempotencyKey, envelope.run_id, envelope.author.name, envelope.competition, envelope.puzzle_type, envelope.puzzle_id, "received", key, timestamp, timestamp).run();
  if (insert.meta.changes !== 1) {
    const winner = await findByIdempotency(env.RESULTS_DB, idempotencyKey);
    if (winner) return asReceipt(winner);
    throw new SafeIngestError("raw_object_conflict");
  }

  try {
    await env.VALIDATE_QUEUE.send({ submission_id: submissionId });
    await transition(env.RESULTS_DB, submissionId, ["received"], "queued");
    return { submission_id: submissionId, idempotency_key: idempotencyKey, state: "queued", duplicate: false };
  } catch {
    await transition(env.RESULTS_DB, submissionId, ["received"], "retryable", { safeError: "queue_unavailable" });
    return { submission_id: submissionId, idempotency_key: idempotencyKey, state: "retryable", duplicate: false };
  }
}
