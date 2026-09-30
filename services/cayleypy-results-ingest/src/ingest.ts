import { validateIngressBatch } from "./ingress-schema.js";
import type { GitHubWriterNamespace } from "./consumer.js";
import { resolveIngestMode } from "./mode.js";
import { MAX_SERIALIZED_BATCH_BYTES } from "./schema.js";
import { validateVersionedEnvelope, type ResultEnvelope, type SchemaVersion } from "./schema-dispatch.js";
import { SafeIngestError, receiveEnvelope, receiveEnvelopeStoreOnly, type IngestEnv } from "./storage.js";
export interface IngestRateLimit {
  limit(input: { key: string }): Promise<{ success: boolean }>;
}

export interface WorkerEnv extends IngestEnv {
  INGEST_MODE?: string;
  INGEST_RATE_LIMIT?: IngestRateLimit;
  GITHUB_WRITER?: GitHubWriterNamespace;
}

export const MAX_REQUEST_BYTES = MAX_SERIALIZED_BATCH_BYTES;
export const MAX_ARCHIVE_REQUEST_BYTES = 32 * 1024 * 1024;
export const MAX_RESULTS_PER_REQUEST = 100;
export const MAX_DECOMPRESSED_ARCHIVE_BYTES = 4 * 1024 * 1024;
export const PER_IP_REQUESTS_PER_MINUTE = 30;
export const PER_IP_STATUS_REQUESTS_PER_MINUTE = 30;
/** Fixed D1 scope cardinality; distinct public IPs can share a status budget. */
export const STATUS_RATE_BUCKET_COUNT = 256;
export const GLOBAL_ENVELOPES_PER_MINUTE = 2_000;
export const RECOVERY_STALE_MS = 60_000;
export const RECOVERY_LIMIT = 50;
const RATE_WINDOW_MS = 60_000;
const RETRY_AFTER_SECONDS = 60;
const ENVELOPE_CONCURRENCY = 2;
export const BODY_READ_TIMEOUT_MS = 15_000;
const DUPLICATE_REREAD_BUDGET = 400;

export class SafeHttpError extends Error {
  constructor(readonly status: number, readonly code: string) {
    super(code);
  }
}

export function jsonResponse(value: unknown, status = 200, headers: HeadersInit = {}): Response {
  const responseHeaders = new Headers(headers);
  responseHeaders.set("content-type", "application/json; charset=utf-8");
  responseHeaders.set("cache-control", "no-store");
  return new Response(JSON.stringify(value), { status, headers: responseHeaders });
}

export function errorResponse(status: number, code: string, headers: HeadersInit = {}): Response {
  return jsonResponse({ error: code }, status, headers);
}

export function methodNotAllowed(allow: "GET" | "POST"): Response {
  return errorResponse(405, "method_not_allowed", { Allow: allow });
}

export function rateLimited(): Response {
  return errorResponse(429, "rate_limited", { "Retry-After": String(RETRY_AFTER_SECONDS) });
}

export function mediaType(request: Request): string {
  return (request.headers.get("content-type") ?? "").split(";", 1)[0].trim().toLowerCase();
}

export function declaredBodyLength(request: Request): number | null {
  const raw = request.headers.get("content-length");
  if (raw === null) return null;
  if (!/^\d+$/.test(raw)) throw new SafeHttpError(400, "invalid_content_length");
  const length = Number(raw);
  if (!Number.isSafeInteger(length)) throw new SafeHttpError(413, "request_too_large");
  return length;
}

async function readBoundedText(
  request: Request,
  maxBytes = MAX_REQUEST_BYTES,
  stream: ReadableStream<Uint8Array> | null = request.body,
  declared: number | null = declaredBodyLength(request),
): Promise<{ text: string; rawByteLength: number }> {
  if (declared !== null && declared > maxBytes) {
    throw new SafeHttpError(413, "request_too_large");
  }

  if (stream === null) return { text: "", rawByteLength: 0 };
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let text = "";
  let total = 0;
  const deadline = Date.now() + BODY_READ_TIMEOUT_MS;
  try {
    while (true) {
      let chunk: ReadableStreamReadResult<Uint8Array>;
      try {
        if (Date.now() >= deadline) throw new SafeHttpError(408, "body_read_timeout");
        let timer: ReturnType<typeof setTimeout> | undefined;
        try {
          chunk = await Promise.race([
            reader.read(),
            new Promise<never>((_resolve, reject) => { timer = setTimeout(() => reject(new SafeHttpError(408, "body_read_timeout")), Math.max(0, deadline - Date.now())); }),
          ]);
        } finally { if (timer !== undefined) clearTimeout(timer); }
      } catch (error) {
        if (error instanceof SafeHttpError) throw error;
        throw new SafeHttpError(400, "invalid_body");
      }
      if (Date.now() >= deadline) throw new SafeHttpError(408, "body_read_timeout");
      if (chunk.done) break;
      total += chunk.value.byteLength;
      if (total > maxBytes) {
        void reader.cancel().catch(() => undefined);
        throw new SafeHttpError(413, "request_too_large");
      }
      try {
        text += decoder.decode(chunk.value, { stream: true });
      } catch {
        void reader.cancel().catch(() => undefined);
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
    void reader.cancel().catch(() => undefined);
    throw error;
  } finally {
    reader.releaseLock();
  }
}

async function parseBatch(request: Request, version: SchemaVersion): Promise<{ value: ResultEnvelope[]; rawByteLength: number } | Response> {
  const type = mediaType(request);
  if (type !== "application/json" && type !== "application/gzip") {
    return errorResponse(415, "unsupported_media_type");
  }

  let body: { text: string; rawByteLength: number };
  let compressedLimitExceeded = false;
  try {
    if (type === "application/json") {
      body = await readBoundedText(request);
    } else {
      const declared = declaredBodyLength(request);
      if (declared !== null && declared > MAX_ARCHIVE_REQUEST_BYTES) {
        throw new SafeHttpError(413, "request_too_large");
      }
      if (request.body === null) {
        body = { text: "", rawByteLength: 0 };
      } else {
        let compressedBytes = 0;
        const bounded = request.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
          transform(chunk, controller) {
            compressedBytes += chunk.byteLength;
            if (compressedBytes > MAX_ARCHIVE_REQUEST_BYTES) { compressedLimitExceeded = true; throw new SafeHttpError(413, "request_too_large"); }
            controller.enqueue(chunk);
          },
        }));
        const decompressed = bounded.pipeThrough(new DecompressionStream("gzip"));
        body = await readBoundedText(request, MAX_DECOMPRESSED_ARCHIVE_BYTES, decompressed, null);
      }
    }
  } catch (error) {
    if (compressedLimitExceeded) return errorResponse(413, "request_too_large");
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
  if (parsed !== null && typeof parsed === "object" && Array.isArray((parsed as {results?:unknown}).results)
      && (parsed as {results:unknown[]}).results.length > MAX_RESULTS_PER_REQUEST) return errorResponse(413, "too_many_results");
  const validation = validateIngressBatch(parsed, version);
  if (!validation.ok) {
    return jsonResponse({ error: "invalid_schema", errors: validation.errors }, 400);
  }
  if (validation.value.results.length > MAX_RESULTS_PER_REQUEST) return errorResponse(413, "too_many_results");
  const integrityAll = [];
  for (const envelope of validation.value.results) integrityAll.push(await validateVersionedEnvelope(envelope));
  const integrityErrors = integrityAll.flatMap((items, index) => items.map((item) => ({ path: `/results/${index}${item.path}`, keyword: item.keyword })));
  if (integrityErrors.length !== 0) {
    return jsonResponse({ error: "invalid_schema", errors: integrityErrors }, 400);
  }
  return { value: validation.value.results, rawByteLength: body.rawByteLength };
}
export async function consumeD1Limit(
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

export async function allowIpRequest(request: Request, env: WorkerEnv): Promise<boolean> {
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
export async function handleIngestBody(request: Request, env: WorkerEnv, version: SchemaVersion): Promise<Response> {
  const mode = resolveIngestMode(env.INGEST_MODE);
  if (mode === "reject") return errorResponse(503, "ingest_disabled");

  const parsed = await parseBatch(request, version);
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

