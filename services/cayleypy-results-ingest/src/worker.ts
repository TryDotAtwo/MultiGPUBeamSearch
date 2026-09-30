import { forwardIngest } from "./ingest-transport.js";
import { consumeValidationMessage } from "./consumer.js";
export { resolveIngestMode, type IngestMode } from "./mode.js";
export { GitHubWriter } from "./github-writer.js";
import { findBySubmissionId } from "./db.js";
import { resolveIngestMode } from "./mode.js";
import type { SchemaVersion } from "./schema-dispatch.js";
import { SafeHttpError, jsonResponse, errorResponse, methodNotAllowed, rateLimited, consumeD1Limit, allowIpRequest, declaredBodyLength, mediaType,
 MAX_REQUEST_BYTES, MAX_ARCHIVE_REQUEST_BYTES, MAX_DECOMPRESSED_ARCHIVE_BYTES, MAX_RESULTS_PER_REQUEST,
 PER_IP_REQUESTS_PER_MINUTE, PER_IP_STATUS_REQUESTS_PER_MINUTE, STATUS_RATE_BUCKET_COUNT, GLOBAL_ENVELOPES_PER_MINUTE, RECOVERY_STALE_MS, RECOVERY_LIMIT, type WorkerEnv } from "./ingest.js";
export * from "./ingest.js";
export const INGRESS_OBJECT_NAME = "ingest-validation-v1";
async function handlePostResults(request: Request, env: WorkerEnv, _version: SchemaVersion): Promise<Response> {
 if (resolveIngestMode(env.INGEST_MODE) === "reject") return errorResponse(503, "ingest_disabled");
 const type = mediaType(request);
 if (type !== "application/json" && type !== "application/gzip") return errorResponse(415, "unsupported_media_type");
 try {
  const length = declaredBodyLength(request);
  if (length !== null && length > (type === "application/gzip" ? MAX_ARCHIVE_REQUEST_BYTES : MAX_REQUEST_BYTES)) return errorResponse(413, "request_too_large");
 } catch (error) { return error instanceof SafeHttpError ? errorResponse(error.status, error.code) : errorResponse(400, "invalid_content_length"); }
 try { if (!(await allowIpRequest(request, env))) return rateLimited(); }
 catch { return errorResponse(503, "rate_limit_unavailable"); }
 try {
  const stub = env.GITHUB_WRITER?.getByName(INGRESS_OBJECT_NAME);
  if (!stub?.fetch) return errorResponse(503, "ingest_unavailable");
  // No RPC JSON, cloning/teeing, body decoding, or receipt parsing at the edge.
  return await forwardIngest(request, input => stub.fetch!(input));
 } catch { return errorResponse(503, "ingest_unavailable", { "Retry-After": "60" }); }
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
      max_archive_request_bytes: MAX_ARCHIVE_REQUEST_BYTES,
      max_decompressed_archive_bytes: MAX_DECOMPRESSED_ARCHIVE_BYTES,
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
    return handlePostResults(request, env, 1);
  }
  if (pathname === "/v2/results") {
    if (request.method !== "POST") return methodNotAllowed("POST");
    return handlePostResults(request, env, 2);
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
  const target = env.GITHUB_WRITER?.getByName("cayleypy-results-v1");
  if (!target?.maintain) throw new Error("maintenance_unavailable");
  await target.maintain(controller.scheduledTime);
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
