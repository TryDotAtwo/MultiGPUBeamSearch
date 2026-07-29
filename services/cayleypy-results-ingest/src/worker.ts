import { findBySubmissionId } from "./db.js";
import { validateBatch, type ResultEnvelopeV1 } from "./schema.js";
import {
  SafeIngestError,
  receiveEnvelope,
  receiveEnvelopeStoreOnly,
  recoverStaleSubmissions,
  type IngestEnv,
} from "./storage.js";

export type IngestMode = "normal" | "store_only" | "reject";

export interface IngestRateLimit {
  limit(input: { key: string }): Promise<{ success: boolean }>;
}

export interface WorkerEnv extends IngestEnv {
  INGEST_MODE?: string;
  INGEST_RATE_LIMIT?: IngestRateLimit;
}

export const MAX_REQUEST_BYTES = 25 * 1024 * 1024;
export const MAX_RESULTS_PER_REQUEST = 100;
export const PER_IP_REQUESTS_PER_MINUTE = 30;
export const GLOBAL_ENVELOPES_PER_MINUTE = 2_000;
export const RECOVERY_STALE_MS = 60_000;
export const RECOVERY_LIMIT = 50;
const RATE_WINDOW_MS = 60_000;
const RETRY_AFTER_SECONDS = 60;

class SafeHttpError extends Error {
  constructor(readonly status: number, readonly code: string) {
    super(code);
  }
}

export function resolveIngestMode(value: unknown): IngestMode {
  if (value === "normal" || value === "store_only" || value === "reject") return value;
  return "reject";
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

async function readBoundedBody(request: Request): Promise<Uint8Array> {
  const declared = declaredBodyLength(request);
  if (declared !== null && declared > MAX_REQUEST_BYTES) {
    throw new SafeHttpError(413, "request_too_large");
  }

  if (request.body === null) return new Uint8Array();
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > MAX_REQUEST_BYTES) {
        await reader.cancel();
        throw new SafeHttpError(413, "request_too_large");
      }
      chunks.push(value);
    }
  } catch (error) {
    if (error instanceof SafeHttpError) throw error;
    throw new SafeHttpError(400, "invalid_body");
  }

  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function parseBatch(request: Request): Promise<{ value: ResultEnvelopeV1[]; rawByteLength: number } | Response> {
  if (mediaType(request) !== "application/json") {
    return errorResponse(415, "unsupported_media_type");
  }

  let bytes: Uint8Array;
  try {
    bytes = await readBoundedBody(request);
  } catch (error) {
    if (error instanceof SafeHttpError) return errorResponse(error.status, error.code);
    return errorResponse(400, "invalid_body");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    return errorResponse(400, "invalid_json");
  }
  const validation = validateBatch(parsed, bytes.byteLength);
  if (!validation.ok) {
    return jsonResponse({ error: "invalid_schema", errors: validation.errors }, 400);
  }
  return { value: validation.value.results, rawByteLength: bytes.byteLength };
}

async function consumeD1Limit(
  db: D1Database,
  scope: string,
  amount: number,
  maximum: number,
  windowStart: number,
): Promise<boolean> {
  const result = await db.prepare(
    `INSERT INTO ingest_rate_limits (scope,window_start,count) VALUES (?,?,?)
     ON CONFLICT(scope) DO UPDATE SET
       window_start = excluded.window_start,
       count = CASE
         WHEN ingest_rate_limits.window_start = excluded.window_start
         THEN ingest_rate_limits.count + excluded.count
         ELSE excluded.count
       END
     WHERE CASE
       WHEN ingest_rate_limits.window_start = excluded.window_start
       THEN ingest_rate_limits.count + excluded.count
       ELSE excluded.count
     END <= ?`,
  ).bind(scope, windowStart, amount, maximum).run();
  return result.meta.changes === 1;
}

async function allowIpRequest(request: Request, env: WorkerEnv, windowStart: number): Promise<boolean> {
  const ip = request.headers.get("CF-Connecting-IP")?.trim() || "unknown";
  if (env.INGEST_RATE_LIMIT !== undefined) {
    try {
      if (!(await env.INGEST_RATE_LIMIT.limit({ key: `ip:${ip}` })).success) return false;
    } catch {
      // The D1 counter below remains the authoritative fallback.
    }
  }
  return consumeD1Limit(env.RESULTS_DB, `ip:${ip}`, 1, PER_IP_REQUESTS_PER_MINUTE, windowStart);
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

async function handlePostResults(request: Request, env: WorkerEnv): Promise<Response> {
  const mode = resolveIngestMode(env.INGEST_MODE);
  if (mode === "reject") return errorResponse(503, "ingest_disabled");

  const windowStart = Math.floor(Date.now() / RATE_WINDOW_MS) * RATE_WINDOW_MS;
  let ipAllowed: boolean;
  try {
    ipAllowed = await allowIpRequest(request, env, windowStart);
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
      windowStart,
    );
  } catch {
    return errorResponse(503, "rate_limit_unavailable");
  }
  if (!globallyAllowed) return rateLimited();

  const outcomes = await Promise.all(parsed.value.map(async (envelope, index) => {
    try {
      const receipt = mode === "normal"
        ? await receiveEnvelope(env, envelope)
        : await receiveEnvelopeStoreOnly(env, envelope);
      return { ok: true as const, receipt: receiptResponse(request, receipt) };
    } catch (error) {
      return { ok: false as const, error: { index, code: safeIngestCode(error) } };
    }
  }));

  const receipts = outcomes.filter((outcome) => outcome.ok).map((outcome) => outcome.receipt);
  const errors = outcomes.filter((outcome) => !outcome.ok).map((outcome) => outcome.error);
  return jsonResponse(errors.length === 0 ? { receipts } : { receipts, errors }, 202);
}

async function handleStatus(submissionId: string, env: WorkerEnv): Promise<Response> {
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
    return handleStatus(decodeURIComponent(statusMatch[1]), env);
  }
  return errorResponse(404, "not_found");
}

export async function scheduled(
  controller: ScheduledController,
  env: WorkerEnv,
  _ctx: ExecutionContext,
): Promise<void> {
  if (resolveIngestMode(env.INGEST_MODE) !== "normal") return;
  await recoverStaleSubmissions(env, {
    staleBefore: new Date(controller.scheduledTime - RECOVERY_STALE_MS),
    limit: RECOVERY_LIMIT,
  });
}

export default { fetch: fetchRequest, scheduled } satisfies ExportedHandler<WorkerEnv>;
