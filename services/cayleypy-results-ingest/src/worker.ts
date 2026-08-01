import { consumeValidationMessage } from "./consumer.js";
export { resolveIngestMode, type IngestMode } from "./mode.js";
import { resolveIngestMode, type IngestMode } from "./mode.js";
export { GitHubWriter } from "./github-writer.js";
import { deleteStagedSubmission, findBySubmissionId, findStagedSubmissions } from "./db.js";
import { MAX_SERIALIZED_BATCH_BYTES, validateBatch, validateBatchIntegrity, type ResultEnvelopeV1 } from "./schema.js";
import {
  SafeIngestError,
  receiveEnvelope,
  receiveEnvelopeStoreOnly,
  recoverStaleSubmissions,
  type IngestEnv,
} from "./storage.js";

export interface IngestRateLimit {
  limit(input: { key: string }): Promise<{ success: boolean }>;
}

export interface WorkerEnv extends IngestEnv {
  INGEST_MODE?: string;
  INGEST_RATE_LIMIT?: IngestRateLimit;
}

export const MAX_REQUEST_BYTES = MAX_SERIALIZED_BATCH_BYTES;
export const MAX_RESULTS_PER_REQUEST = 100;
export const PER_IP_REQUESTS_PER_MINUTE = 30;
export const PER_IP_STATUS_REQUESTS_PER_MINUTE = 30;
/** Fixed D1 scope cardinality; distinct public IPs can share a status budget. */
export const STATUS_RATE_BUCKET_COUNT = 256;
export const GLOBAL_ENVELOPES_PER_MINUTE = 2_000;
export const RECOVERY_STALE_MS = 60_000;
export const RECOVERY_LIMIT = 50;
const RATE_WINDOW_MS = 60_000;
const RETRY_AFTER_SECONDS = 60;
const ENVELOPE_CONCURRENCY = 8;
const DUPLICATE_REREAD_BUDGET = 400;

class SafeHttpError extends Error {
  constructor(readonly status: number, readonly code: string) {
    super(code);
  }
}

function jsonResponse(value: unknown, status = 200, headers: HeadersInit = {}): Response {
  const responseHeaders = new Headers(headers);
  responseHeaders.set("content-type", "application/json; charset=utf-8");
  responseHeaders.set("cache-control", "no-store");
  return new Response(JSON.stringify(value), { status, headers: responseHeaders });
}

function errorResponse(status: number, code: string, headers: HeadersInit = {}): Response {
  return jsonResponse({ error: code }, status, headers);
}

function methodNotAllowed(allow: "GET" | "POST"): Response {
  return errorResponse(405, "method_not_allowed", { Allow: allow });
}

function rateLimited(): Response {
  return errorResponse(429, "rate_limited", { "Retry-After": String(RETRY_AFTER_SECONDS) });
}

function mediaType(request: Request): string {
  return (request.headers.get("content-type") ?? "").split(";", 1)[0].trim().toLowerCase();
}

function declaredBodyLength(request: Request): number | null {
  const raw = request.headers.get("content-length");
  if (raw === null) return null;
  if (!/^\d+$/.test(raw)) throw new SafeHttpError(400, "invalid_content_length");
  const length = Number(raw);
  if (!Number.isSafeInteger(length)) throw new SafeHttpError(413, "request_too_large");
  return length;
}

async function readBoundedText(request: Request): Promise<{ text: string; rawByteLength: number }> {
  const declared = declaredBodyLength(request);
  if (declared !== null && declared > MAX_REQUEST_BYTES) {
    throw new SafeHttpError(413, "request_too_large");
  }

  if (request.body === null) return { text: "", rawByteLength: 0 };
  const reader = request.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let text = "";
  let total = 0;
  try {
    while (true) {
      let chunk: ReadableStreamReadResult<Uint8Array>;
      try {
        chunk = await reader.read();
      } catch {
        throw new SafeHttpError(400, "invalid_body");
      }
      if (chunk.done) break;
      total += chunk.value.byteLength;
      if (total > MAX_REQUEST_BYTES) {
        try { await reader.cancel(); } catch { /* Best-effort cancellation after the hard bound. */ }
        throw new SafeHttpError(413, "request_too_large");
      }
      try {
        text += decoder.decode(chunk.value, { stream: true });
      } catch {
        try { await reader.cancel(); } catch { /* Best-effort cancellation after invalid UTF-8. */ }
        throw new SafeHttpError(400, "invalid_json");
      }
    }
    try {
      text += decoder.decode();
    } catch {
      throw new SafeHttpError(400, "invalid_json");
    }
    return { text, rawByteLength: total };
  } catch (error) {
    text = "";
    throw error;
  } finally {
    reader.releaseLock();
  }
}

async function parseBatch(request: Request): Promise<{ value: ResultEnvelopeV1[]; rawByteLength: number } | Response> {
  if (mediaType(request) !== "application/json") {
    return errorResponse(415, "unsupported_media_type");
  }

  let body: { text: string; rawByteLength: number };
  try {
    body = await readBoundedText(request);
  } catch (error) {
    if (error instanceof SafeHttpError) return errorResponse(error.status, error.code);
    return errorResponse(400, "invalid_body");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(body.text);
  } catch {
    return errorResponse(400, "invalid_json");
  } finally {
    body.text = "";
  }
  const validation = validateBatch(parsed, body.rawByteLength);
  if (!validation.ok) {
    return jsonResponse({ error: "invalid_schema", errors: validation.errors }, 400);
  }
  const integrityErrors = await validateBatchIntegrity(validation.value.results);
  if (integrityErrors.length !== 0) {
    return jsonResponse({ error: "invalid_schema", errors: integrityErrors }, 400);
  }
  return { value: validation.value.results, rawByteLength: body.rawByteLength };
}
async function consumeD1Limit(
  db: D1Database,
  scope: string,
  amount: number,
  maximum: number,
): Promise<boolean> {
  const windowStart = Math.floor(Date.now() / RATE_WINDOW_MS) * RATE_WINDOW_MS;
  const result = await db.prepare(
    `INSERT INTO ingest_rate_limits (scope,window_start,count) VALUES (?,?,?)
     ON CONFLICT(scope) DO UPDATE SET
       window_start = excluded.window_start,
       count = CASE
         WHEN ingest_rate_limits.window_start = excluded.window_start
         THEN ingest_rate_limits.count + excluded.count
         ELSE excluded.count
       END
     WHERE
       (excluded.window_start > ingest_rate_limits.window_start AND excluded.count <= ?)
       OR
       (excluded.window_start = ingest_rate_limits.window_start
        AND ingest_rate_limits.count + excluded.count <= ?)`,
  ).bind(scope, windowStart, amount, maximum, maximum).run();
  return result.meta.changes === 1;
}

async function allowIpRequest(request: Request, env: WorkerEnv): Promise<boolean> {
  const ip = request.headers.get("CF-Connecting-IP")?.trim() || "unknown";
  if (env.INGEST_RATE_LIMIT !== undefined) {
    try {
      if (!(await env.INGEST_RATE_LIMIT.limit({ key: `ip:${ip}` })).success) return false;
    } catch {
      // The D1 counter below remains the authoritative fallback.
    }
  }
  return consumeD1Limit(env.RESULTS_DB, `ip:${ip}`, 1, PER_IP_REQUESTS_PER_MINUTE);
}
function receiptResponse(
  request: Request,
  receipt: { submission_id: string; idempotency_key: string },
): { submission_id: string; idempotency_key: string; status_url: string } {
  return {
    submission_id: receipt.submission_id,
    idempotency_key: receipt.idempotency_key,
    status_url: new URL(`/v1/submissions/${receipt.submission_id}`, request.url).toString(),
  };
}

function safeIngestCode(error: unknown): string {
  return error instanceof SafeIngestError ? error.code : "ingest_failed";
}

async function mapBounded<T, U>(
  values: readonly T[],
  concurrency: number,
  operation: (value: T, index: number) => Promise<U>,
): Promise<U[]> {
  const output = new Array<U>(values.length);
  let nextIndex = 0;
  const consume = async () => {
    while (nextIndex < values.length) {
      const index = nextIndex;
      nextIndex += 1;
      output[index] = await operation(values[index], index);
    }
  };
  await Promise.all(Array.from(
    { length: Math.min(concurrency, values.length) },
    () => consume(),
  ));
  return output;
}
async function handlePostResults(request: Request, env: WorkerEnv): Promise<Response> {
  const mode = resolveIngestMode(env.INGEST_MODE);
  if (mode === "reject") return errorResponse(503, "ingest_disabled");

  let ipAllowed: boolean;
  try {
    ipAllowed = await allowIpRequest(request, env);
  } catch {
    return errorResponse(503, "rate_limit_unavailable");
  }
  if (!ipAllowed) return rateLimited();

  const parsed = await parseBatch(request);
  if (parsed instanceof Response) return parsed;

  let globallyAllowed: boolean;
  try {
    globallyAllowed = await consumeD1Limit(
      env.RESULTS_DB,
      "global",
      parsed.value.length,
      GLOBAL_ENVELOPES_PER_MINUTE,
    );
  } catch {
    return errorResponse(503, "rate_limit_unavailable");
  }
  if (!globallyAllowed) return rateLimited();

  const duplicatePollAttempts = Math.max(
    1,
    Math.floor(DUPLICATE_REREAD_BUDGET / parsed.value.length),
  );
  const outcomes = await mapBounded(parsed.value, ENVELOPE_CONCURRENCY, async (envelope, index) => {
    try {
      const receipt = mode === "normal"
        ? await receiveEnvelope(env, envelope, { duplicatePollAttempts })
        : await receiveEnvelopeStoreOnly(env, envelope);
      return { ok: true as const, receipt: receiptResponse(request, receipt) };
    } catch (error) {
      return { ok: false as const, error: { index, code: safeIngestCode(error) } };
    }
  });

  const receipts = outcomes.filter((outcome) => outcome.ok).map((outcome) => outcome.receipt);
  const errors = outcomes.filter((outcome) => !outcome.ok).map((outcome) => outcome.error);
  return jsonResponse(errors.length === 0 ? { receipts } : { receipts, errors }, 202);
}

export function statusRateScope(ip: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < ip.length; index += 1) {
    hash ^= ip.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return "status-bucket:" + ((hash >>> 0) % STATUS_RATE_BUCKET_COUNT);
}
async function allowStatusRequest(request: Request, env: WorkerEnv): Promise<boolean> {
  const ip = request.headers.get("CF-Connecting-IP")?.trim() || "unknown";
  return consumeD1Limit(env.RESULTS_DB, statusRateScope(ip), 1, PER_IP_STATUS_REQUESTS_PER_MINUTE);
}

async function handleStatus(request: Request, submissionId: string, env: WorkerEnv): Promise<Response> {
  try {
    if (!(await allowStatusRequest(request, env))) return rateLimited();
  } catch {
    return errorResponse(503, "rate_limit_unavailable");
  }
  let row;
  try {
    row = await findBySubmissionId(env.RESULTS_DB, submissionId);
  } catch {
    return errorResponse(503, "status_unavailable");
  }
  if (row === null) return errorResponse(404, "not_found");
  return jsonResponse({
    submission_id: row.submission_id,
    idempotency_key: row.idempotency_key,
    state: row.state,
    safe_error: row.safe_error,
    retry_count: row.retry_count,
    updated_at: row.updated_at,
  });
}

function health(env: WorkerEnv): Response {
  return jsonResponse({
    status: "ok",
    ingest_mode: resolveIngestMode(env.INGEST_MODE),
    limits: {
      max_request_bytes: MAX_REQUEST_BYTES,
      max_results_per_request: MAX_RESULTS_PER_REQUEST,
      per_ip_requests_per_minute: PER_IP_REQUESTS_PER_MINUTE,
      global_envelopes_per_minute: GLOBAL_ENVELOPES_PER_MINUTE,
    },
    recovery: { stale_ms: RECOVERY_STALE_MS, limit: RECOVERY_LIMIT },
  });
}

export async function fetchRequest(
  request: Request,
  env: WorkerEnv,
  _ctx: ExecutionContext,
): Promise<Response> {
  const pathname = new URL(request.url).pathname;
  if (pathname === "/v1/results") {
    if (request.method !== "POST") return methodNotAllowed("POST");
    return handlePostResults(request, env);
  }
  if (pathname === "/healthz") {
    if (request.method !== "GET") return methodNotAllowed("GET");
    return health(env);
  }
  const statusMatch = /^\/v1\/submissions\/([^/]+)$/.exec(pathname);
  if (statusMatch !== null) {
    if (request.method !== "GET") return methodNotAllowed("GET");
    let submissionId: string;
    try {
      submissionId = decodeURIComponent(statusMatch[1]);
    } catch {
      return errorResponse(400, "invalid_submission_id");
    }
    return handleStatus(request, submissionId, env);
  }
  return errorResponse(404, "not_found");
}

export async function scheduled(
  controller: ScheduledController,
  env: WorkerEnv,
  _ctx: ExecutionContext,
): Promise<void> {
  if (resolveIngestMode(env.INGEST_MODE) !== "normal") return;
  for (const row of await findStagedSubmissions(env.RESULTS_DB, 100)) {
    await env.RAW_RESULTS.delete(row.raw_r2_key);
    const deleted = await deleteStagedSubmission(env.RESULTS_DB, row.submission_id);
    if (!deleted && await findBySubmissionId(env.RESULTS_DB, row.submission_id)) {
      throw new Error("staged_cleanup_conflict");
    }
  }
  await recoverStaleSubmissions(env, {
    staleBefore: new Date(controller.scheduledTime - RECOVERY_STALE_MS),
    limit: RECOVERY_LIMIT,
  });
}

export async function queue(
  batch: MessageBatch<unknown>,
  env: WorkerEnv,
  _ctx: ExecutionContext,
): Promise<void> {
  const mode = resolveIngestMode(env.INGEST_MODE);
  for (const message of batch.messages) {
    await consumeValidationMessage(message, env, mode);
  }
}

export default { fetch: fetchRequest, scheduled, queue } satisfies ExportedHandler<WorkerEnv>;
