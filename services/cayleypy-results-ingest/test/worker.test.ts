import { createHash } from "node:crypto";
import { env } from "cloudflare:workers";
import canonicalGolden from "../../../configs/cayleypy_results_v1_golden.json";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { canonicalJson, computeIdempotency } from "../src/ids.js";
import type { ResultEnvelopeV1 } from "../src/schema.js";
import { replayDeadLetters } from "../src/operator-replay.js";

import {
  GLOBAL_ENVELOPES_PER_MINUTE,
  MAX_REQUEST_BYTES,
  MAX_RESULTS_PER_REQUEST,
  PER_IP_REQUESTS_PER_MINUTE,
  RECOVERY_LIMIT,
  RECOVERY_STALE_MS,
  STATUS_RATE_BUCKET_COUNT,
  statusRateScope,
  fetchRequest,
  scheduled,
  queue,
  resolveIngestMode,
  type WorkerEnv,
} from "../src/worker.js";

const URL = "https://ingest.example.test";

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${stableJson(record[key])}`).join(",")}}`;
}
function semanticHash(envelope: ResultEnvelopeV1): string {
  const { client_submission_id: _client, idempotency_key: _key, submitted_at: _at, run_id: _run, ...semantic } = envelope;
  return createHash("sha256").update(stableJson(semantic), "utf8").digest("hex");
}
function validEnvelope(index = 0) {
  const envelope = structuredClone(canonicalGolden.cases[0].envelope) as unknown as ResultEnvelopeV1;
  envelope.client_submission_id = `018f7a24-8f6b-7c8e-9d1b-${(0x2a3b4c5d6e7fn + BigInt(index)).toString(16).padStart(12, "0")}`;
  envelope.run_id = `run-20260729-${index}`;
  envelope.puzzle_id = 42 + index;
  envelope.idempotency_key = semanticHash(envelope);
  return envelope;
}
function bindings(mode: string | undefined = "normal"): WorkerEnv {
  const result: WorkerEnv = {
    RESULTS_DB: env.RESULTS_DB,
    RAW_RESULTS: env.RAW_RESULTS,
    VALIDATE_QUEUE: { send: async () => undefined } as unknown as Queue,
  };
  if (mode !== undefined) result.INGEST_MODE = mode;
  return result;
}

function context(): ExecutionContext {
  return { waitUntil: () => undefined, passThroughOnException: () => undefined } as unknown as ExecutionContext;
}

beforeEach(async () => {
  await env.RESULTS_DB.exec("DELETE FROM submissions");
  const listed = await env.RAW_RESULTS.list({ prefix: "raw/" });
  await env.RESULTS_DB.exec("DELETE FROM ingest_rate_limits");
  if (listed.objects.length) await env.RAW_RESULTS.delete(listed.objects.map((object) => object.key));
});

describe("Task 3 public ingest contract", () => {
  test("resolves only exact case-sensitive modes and fails closed", () => {
    expect(resolveIngestMode("normal")).toBe("normal");
    expect(resolveIngestMode("store_only")).toBe("store_only");
    expect(resolveIngestMode("reject")).toBe("reject");
    for (const value of [undefined, "", "Normal", "STORE_ONLY", " normal", "normal ", "unknown", null]) {
      expect(resolveIngestMode(value)).toBe("reject");
    }
  });

  test("normal POST returns a durable receipt", async () => {
    const request = new Request(`${URL}/v1/results`, {
      method: "POST",
      headers: { "content-type": "application/json", "CF-Connecting-IP": "203.0.113.42" },
      body: JSON.stringify({ schema_version: 1, results: [validEnvelope()] }),
    });
    const response = await fetchRequest(request, bindings(), context());
    expect(response.status).toBe(202);
    expect(await response.json()).toMatchObject({ receipts: [{ status_url: expect.stringContaining("/v1/submissions/") }] });
  });

  test("health exposes exact limits and no raw mode", async () => {
    const response = await fetchRequest(new Request(`${URL}/healthz`), bindings("UNKNOWN_RAW_MODE"), context());
    const text = await response.text();
    expect(text).not.toContain("UNKNOWN_RAW_MODE");
    expect(JSON.parse(text)).toEqual({
      status: "ok", ingest_mode: "reject",
      limits: {
        max_request_bytes: MAX_REQUEST_BYTES,
        max_results_per_request: MAX_RESULTS_PER_REQUEST,
        per_ip_requests_per_minute: PER_IP_REQUESTS_PER_MINUTE,
        global_envelopes_per_minute: GLOBAL_ENVELOPES_PER_MINUTE,
      },
      recovery: { stale_ms: RECOVERY_STALE_MS, limit: RECOVERY_LIMIT },
    });
  });
});

const TEST_IP = "203.0.113.42";

function customBindings(
  mode: string | undefined,
  options: {
    db?: D1Database;
    queue?: Queue;
    bucket?: R2Bucket;
    rateLimit?: { limit(input: { key: string }): Promise<{ success: boolean }> };
  } = {},
): WorkerEnv {
  const value: WorkerEnv = {
    RESULTS_DB: options.db ?? env.RESULTS_DB,
    RAW_RESULTS: options.bucket ?? env.RAW_RESULTS,
    VALIDATE_QUEUE: options.queue ?? ({ send: async () => undefined } as unknown as Queue),
  };
  if (mode !== undefined) value.INGEST_MODE = mode;
  if (options.rateLimit) value.INGEST_RATE_LIMIT = options.rateLimit;
  return value;
}

function resultBatch(...indexes: number[]) {
  return { schema_version: 1 as const, results: indexes.map((index) => validEnvelope(index)) };
}

function uniqueResultBatch(count: number, offset = 0) {
  return { schema_version: 1 as const, results: Array.from({ length: count }, (_, index) => validEnvelope(offset + index)) };
}

function postRequest(body: BodyInit | null, contentType = "application/json", extraHeaders: HeadersInit = {}): Request {
  const headers = new Headers(extraHeaders);
  if (contentType) headers.set("content-type", contentType);
  headers.set("CF-Connecting-IP", TEST_IP);
  return new Request(`${URL}/v1/results`, { method: "POST", headers, body });
}

async function postJson(value: unknown, workerEnv: WorkerEnv): Promise<Response> {
  return fetchRequest(postRequest(JSON.stringify(value)), workerEnv, context());
}

function proxyBucket(overrides: Partial<R2Bucket> = {}): R2Bucket {
  return {
    head: (...args: Parameters<R2Bucket["head"]>) => env.RAW_RESULTS.head(...args),
    get: (...args: Parameters<R2Bucket["get"]>) => env.RAW_RESULTS.get(...args),
    put: (...args: Parameters<R2Bucket["put"]>) => env.RAW_RESULTS.put(...args),
    delete: (...args: Parameters<R2Bucket["delete"]>) => env.RAW_RESULTS.delete(...args),
    list: (...args: Parameters<R2Bucket["list"]>) => env.RAW_RESULTS.list(...args),
    ...overrides,
  } as unknown as R2Bucket;
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

type D1TerminalMethod = "all" | "first" | "raw" | "run";

function proxyDatabase(
  beforeTerminal: (sql: string, method: D1TerminalMethod, values: unknown[]) => void | Promise<void>,
): D1Database {
  const wrap = (statement: D1PreparedStatement, sql: string, values: unknown[] = []): D1PreparedStatement =>
    new Proxy(statement, {
      get(target, property) {
        if (property === "bind") {
          return (...nextValues: unknown[]) => wrap(target.bind(...nextValues), sql, nextValues);
        }
        if (property === "all" || property === "first" || property === "raw" || property === "run") {
          return async (...args: unknown[]) => {
            await beforeTerminal(sql, property, values);
            return Reflect.apply(
              Reflect.get(target, property, target) as (...input: unknown[]) => unknown,
              target,
              args,
            );
          };
        }
        const member = Reflect.get(target, property, target) as unknown;
        return typeof member === "function" ? member.bind(target) : member;
      },
    });
  return new Proxy(env.RESULTS_DB, {
    get(target, property) {
      if (property === "prepare") return (sql: string) => wrap(target.prepare(sql), sql);
      const member = Reflect.get(target, property, target) as unknown;
      return typeof member === "function" ? member.bind(target) : member;
    },
  });
}

function countedBucket(onCall: () => void): R2Bucket {
  return new Proxy(env.RAW_RESULTS, {
    get(target, property) {
      const member = Reflect.get(target, property, target) as unknown;
      if (typeof member !== "function") return member;
      return (...args: unknown[]) => {
        onCall();
        return Reflect.apply(member as (...input: unknown[]) => unknown, target, args);
      };
    },
  });
}

async function rowCount(): Promise<number> {
  return (await env.RESULTS_DB.prepare("SELECT COUNT(*) AS count FROM submissions").first<number>("count")) ?? -1;
}

async function rawCount(): Promise<number> {
  return (await env.RAW_RESULTS.list({ prefix: "raw/" })).objects.length;
}

function controller(scheduledTime = Date.now()): ScheduledController {
  return { scheduledTime, cron: "*/1 * * * *", noRetry: () => undefined } as ScheduledController;
}

async function seedSubmission(
  index: number,
  state: "received" | "queued" | "retryable" | "dead_letter",
  updatedAt: string,
  safeError: string | null = "stale_error",
  retryCount = 0,
): Promise<string> {
  const submissionId = `019c2000-0000-7000-8000-${index.toString().padStart(12, "0")}`;
  const rawKey = `raw/v1/2000/01/01/${submissionId}.json`;
  const envelope = validEnvelope(index + 1000);
  await env.RAW_RESULTS.put(rawKey, canonicalJson(envelope));
  await env.RESULTS_DB.prepare(
    "INSERT INTO submissions (submission_id,idempotency_key,run_id,author_name,competition,puzzle_type,puzzle_id,state,raw_r2_key,safe_error,retry_count,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
  ).bind(
    submissionId,
    index.toString(16).padStart(64, "0"),
    envelope.run_id,
    envelope.author.name,
    envelope.competition,
    envelope.puzzle_type,
    envelope.puzzle_id,
    state,
    rawKey,
    safeError,
    retryCount,
    "2000-01-01T00:00:00.000Z",
    updatedAt,
  ).run();
  return submissionId;
}

describe("fail-closed modes and bounded request parsing", () => {
  test.each(["reject", undefined, "", "Normal", "unknown"])(
    "mode %s writes nothing and does not touch the rate binding",
    async (mode) => {
      let rateCalls = 0;
      let queueCalls = 0;
      const response = await postJson(resultBatch(0), customBindings(mode, {
        queue: { send: async () => { queueCalls += 1; } } as unknown as Queue,
        rateLimit: { limit: async () => { rateCalls += 1; return { success: true }; } },
      }));
      expect(response.status).toBe(503);
      expect(await response.json()).toEqual({ error: "ingest_disabled" });
      expect({ rateCalls, queueCalls, rows: await rowCount(), raw: await rawCount() }).toEqual({
        rateCalls: 0, queueCalls: 0, rows: 0, raw: 0,
      });
      expect(await env.RESULTS_DB.prepare("SELECT COUNT(*) AS count FROM ingest_rate_limits").first<number>("count")).toBe(0);
    },
  );

  test("rejects the wrong method with an Allow header", async () => {
    const response = await fetchRequest(new Request(`${URL}/v1/results`, { method: "GET" }), bindings(), context());
    expect(response.status).toBe(405);
    expect(response.headers.get("allow")).toBe("POST");
    expect(await response.json()).toEqual({ error: "method_not_allowed" });
  });

  test.each(["", "text/plain", "application/problem+json"])("rejects content type %s", async (contentType) => {
    const response = await fetchRequest(postRequest(JSON.stringify(resultBatch(0)), contentType), bindings(), context());
    expect(response.status).toBe(415);
    expect(await response.json()).toEqual({ error: "unsupported_media_type" });
    expect(await rowCount()).toBe(0);
  });

  test("accepts application/json parameters", async () => {
    const response = await fetchRequest(postRequest(JSON.stringify(resultBatch(0)), "Application/JSON; charset=utf-8"), bindings(), context());
    expect(response.status).toBe(202);
  });

  test("rejects an oversized Content-Length before reading the stream", async () => {
    const stream = new ReadableStream<Uint8Array>({ pull(streamController) { streamController.enqueue(new Uint8Array([0])); } });
    const request = postRequest(stream, "application/json", { "content-length": String(MAX_REQUEST_BYTES + 1) });
    const response = await fetchRequest(
      request,
      bindings(),
      context(),
    );
    expect(response.status).toBe(413);
    expect(await response.json()).toEqual({ error: "request_too_large" });
    expect(request.bodyUsed).toBe(false);
  });

  test("counts a streaming body and stops immediately above the four MiB hard bound", async () => {
    let chunks = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(streamController) {
        chunks += 1;
        streamController.enqueue(new Uint8Array(chunks <= 4 ? 1024 * 1024 : 1));
        if (chunks === 5) streamController.close();
      },
    });
    const response = await fetchRequest(postRequest(stream), bindings(), context());
    expect(response.status).toBe(413);
    expect(await response.json()).toEqual({ error: "request_too_large" });
    expect(chunks).toBe(5);
    expect(await rowCount()).toBe(0);
  });

  test("accepts an exact four MiB chunked JSON body with UTF-8 split across chunks", async () => {
    const envelope = { ...validEnvelope(0), author: { name: "Ada-😀", verification: "claimed" as const } } as ResultEnvelopeV1;
    envelope.idempotency_key = semanticHash(envelope);
    const encoded = new TextEncoder().encode(JSON.stringify({ schema_version: 1, results: [envelope] }));
    const body = new Uint8Array(4 * 1024 * 1024);
    body.fill(0x20);
    body.set(encoded);
    const emojiStart = body.indexOf(0xf0);
    expect(emojiStart).toBeGreaterThan(0);
    const ends = [emojiStart + 2, 1024 * 1024, 2 * 1024 * 1024, 3 * 1024 * 1024, body.byteLength];
    let start = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        const end = ends.shift();
        if (end === undefined) return controller.close();
        controller.enqueue(body.slice(start, end));
        start = end;
      },
    });
    const response = await fetchRequest(postRequest(stream), customBindings("store_only"), context());
    expect(response.status).toBe(202);
    expect((await response.json() as { receipts: unknown[] }).receipts).toHaveLength(1);
  });

  test("returns value-free malformed JSON and strict-schema errors", async () => {
    const malformed = await fetchRequest(postRequest('{"secret":"DO_NOT_ECHO"'), bindings(), context());
    expect(malformed.status).toBe(400);
    expect(await malformed.json()).toEqual({ error: "invalid_json" });
    const schemaResponse = await postJson({ schema_version: 1, results: [{ secret: "DO_NOT_ECHO" }] }, bindings());
    expect(schemaResponse.status).toBe(400);
    const text = await schemaResponse.text();
    expect(text).not.toContain("DO_NOT_ECHO");
    const parsed = JSON.parse(text) as { error: string; errors: Array<Record<string, unknown>> };
    expect(parsed.error).toBe("invalid_schema");
    expect(parsed.errors.length).toBeGreaterThan(0);
    expect(parsed.errors.every((error) => Object.keys(error).sort().join(",") === "keyword,path")).toBe(true);
  });

  test("enforces at most 100 envelopes", async () => {
    const response = await postJson(resultBatch(...Array.from({ length: 101 }, (_, index) => index)), bindings());
    expect(response.status).toBe(400);
    expect(await response.json()).toMatchObject({ error: "invalid_schema" });
    expect(await rowCount()).toBe(0);
  });
});

describe("durable receipts, status, health, and logging safety", () => {
  test("normal mode persists each envelope, sends per-id work, and gives duplicate receipts", async () => {
    const messages: unknown[] = [];
    const workerEnv = customBindings("normal", {
      queue: { send: async (message: unknown) => { messages.push(message); } } as unknown as Queue,
    });
    const first = await postJson(resultBatch(0, 1), workerEnv);
    expect(first.status).toBe(202);
    const body = await first.json() as {
      receipts: Array<{ submission_id: string; idempotency_key: string; status_url: string }>;
    };
    expect(body.receipts).toHaveLength(2);
    expect(messages).toEqual(body.receipts.map(({ submission_id }) => ({ submission_id })));
    expect(body.receipts.every((receipt) => receipt.status_url === `${URL}/v1/submissions/${receipt.submission_id}`)).toBe(true);
    expect({ rows: await rowCount(), raw: await rawCount() }).toEqual({ rows: 2, raw: 2 });

    const duplicate = await postJson(resultBatch(0), workerEnv);
    expect(duplicate.status).toBe(202);
    expect((await duplicate.json() as { receipts: typeof body.receipts }).receipts[0]).toEqual(body.receipts[0]);
    expect(messages).toHaveLength(2);
    expect({ rows: await rowCount(), raw: await rawCount() }).toEqual({ rows: 2, raw: 2 });
  });

  test("returns mixed durable receipt states when one per-item Queue send fails", async () => {
    const queue = {
      send: async ({ submission_id }: { submission_id: string }) => {
        const puzzleId = await env.RESULTS_DB.prepare("SELECT puzzle_id FROM submissions WHERE submission_id = ?")
          .bind(submission_id).first<number>("puzzle_id");
        if (puzzleId === 43) throw new Error("QUEUE_SECRET_DETAIL");
      },
    } as unknown as Queue;
    const response = await postJson(resultBatch(0, 1), customBindings("normal", { queue }));
    expect(response.status).toBe(202);
    const text = await response.text();
    expect(text).not.toContain("QUEUE_SECRET_DETAIL");
    expect(text).not.toContain("puzzle_id");
    expect((JSON.parse(text) as { receipts: unknown[] }).receipts).toHaveLength(2);
    const rows = await env.RESULTS_DB.prepare("SELECT puzzle_id,state,safe_error FROM submissions ORDER BY puzzle_id")
      .all<{ puzzle_id: number; state: string; safe_error: string | null }>();
    expect(rows.results).toEqual([
      { puzzle_id: 42, state: "queued", safe_error: null },
      { puzzle_id: 43, state: "retryable", safe_error: "queue_unavailable" },
    ]);
    expect(await rawCount()).toBe(2);
  });

  test("returns 202 with safe per-index errors when one item cannot be persisted", async () => {
    const bucket = proxyBucket({
      put: async (key, value, options) => {
        const text = typeof value === "string" ? value : "";
        if (text.includes('"puzzle_id":43')) throw new Error("R2_SECRET_DETAIL");
        return env.RAW_RESULTS.put(key, value, options);
      },
    });
    const response = await postJson(resultBatch(0, 1), customBindings("normal", { bucket }));
    expect(response.status).toBe(202);
    const text = await response.text();
    expect(text).not.toContain("R2_SECRET_DETAIL");
    expect(text).not.toContain("puzzle_id");
    const body = JSON.parse(text) as { receipts: unknown[]; errors: Array<{ index: number; code: string }> };
    expect(body.receipts).toHaveLength(1);
    expect(body.errors).toEqual([{ index: 1, code: "ingest_failed" }]);
    expect(await rowCount()).toBe(1);
  });

  test("store_only persists immutable raw and one received receipt without Queue sends", async () => {
    let sends = 0;
    const workerEnv = customBindings("store_only", {
      queue: { send: async () => { sends += 1; } } as unknown as Queue,
    });
    const first = await postJson(resultBatch(0), workerEnv);
    const duplicate = await postJson(resultBatch(0), workerEnv);
    expect(first.status).toBe(202);
    expect(duplicate.status).toBe(202);
    const firstReceipt = (await first.json() as { receipts: Array<{ submission_id: string }> }).receipts[0];
    const duplicateReceipt = (await duplicate.json() as { receipts: Array<{ submission_id: string }> }).receipts[0];
    expect(duplicateReceipt).toEqual(firstReceipt);
    expect({ sends, rows: await rowCount(), raw: await rawCount() }).toEqual({ sends: 0, rows: 1, raw: 1 });
    expect(await env.RESULTS_DB.prepare("SELECT state FROM submissions").first<string>("state")).toBe("received");
  });

  test("GET status returns only the public receipt state and a safe 404", async () => {
    const accepted = await postJson(resultBatch(0), bindings());
    const receipt = (await accepted.json() as { receipts: Array<{ submission_id: string }> }).receipts[0];
    const response = await fetchRequest(new Request(`${URL}/v1/submissions/${receipt.submission_id}`), bindings(), context());
    expect(response.status).toBe(200);
    const body = await response.json() as Record<string, unknown>;
    expect(Object.keys(body).sort()).toEqual([
      "idempotency_key", "retry_count", "safe_error", "state", "submission_id", "updated_at",
    ]);
    expect(body).toMatchObject({ submission_id: receipt.submission_id, state: "queued", safe_error: null, retry_count: 0 });
    expect(JSON.stringify(body)).not.toContain("raw/v1/");

    const missing = await fetchRequest(new Request(`${URL}/v1/submissions/019cffff-ffff-7fff-8fff-ffffffffffff`), bindings(), context());
    expect(missing.status).toBe(404);
    expect(await missing.json()).toEqual({ error: "not_found" });
  });

  test("GET status uses the D1 IP limit and returns only a safe 429", async () => {
    const accepted = await postJson(resultBatch(0), bindings());
    const receipt = (await accepted.json() as { receipts: Array<{ submission_id: string }> }).receipts[0];
    const windowStart = Math.floor(Date.now() / 60_000) * 60_000;
    await env.RESULTS_DB.prepare("INSERT INTO ingest_rate_limits (scope,window_start,count) VALUES (?,?,?)")
      .bind(statusRateScope(TEST_IP), windowStart, 30).run();
    const response = await fetchRequest(
      new Request(`${URL}/v1/submissions/${receipt.submission_id}`, { headers: { "CF-Connecting-IP": TEST_IP } }),
      bindings(), context(),
    );
    expect(response.status).toBe(429);
    expect(response.headers.get("retry-after")).toBe("60");
    const body = await response.text();
    expect(body).toBe('{"error":"rate_limited"}');
    expect(body).not.toContain(receipt.submission_id);
  });

  test("status rate scopes stay within the fixed bucket cardinality for many distinct IPs", async () => {
    const scopes = new Set<string>();
    for (let index = 0; index < STATUS_RATE_BUCKET_COUNT * 3; index += 1) {
      const ip = "198.51." + Math.floor(index / 256) + "." + (index % 256);
      scopes.add(statusRateScope(ip));
      await fetchRequest(
        new Request(URL + "/v1/submissions/019cffff-ffff-7fff-8fff-ffffffffffff", {
          headers: { "CF-Connecting-IP": ip },
        }),
        bindings(),
        context(),
      );
    }
    expect(scopes.size).toBeLessThanOrEqual(STATUS_RATE_BUCKET_COUNT);
    const persisted = await env.RESULTS_DB.prepare(
      "SELECT COUNT(*) AS count FROM ingest_rate_limits WHERE scope LIKE 'status-bucket:%'",
    ).first<number>("count");
    expect(persisted).toBeLessThanOrEqual(STATUS_RATE_BUCKET_COUNT);
  }, 30_000);
  test("operator dead-letter replay is dry-run by default and retains raw before a bounded apply", async () => {
    const first = await seedSubmission(801, "dead_letter", "2000-01-01T00:00:00.000Z", "publisher_unavailable");
    const second = await seedSubmission(802, "dead_letter", "2000-01-02T00:00:00.000Z", "publisher_unavailable");
    const dry = await replayDeadLetters(bindings(), { limit: 1 });
    expect(dry).toEqual({ dry_run: true, selected: [first], replayed: [], skipped_missing_raw: [] });
    expect(await env.RESULTS_DB.prepare("SELECT state FROM submissions WHERE submission_id = ?").bind(first).first<string>("state")).toBe("dead_letter");
    const applied = await replayDeadLetters(bindings(), { apply: true, limit: 1 });
    expect(applied).toEqual({ dry_run: false, selected: [first], replayed: [first], skipped_missing_raw: [] });
    expect(await env.RESULTS_DB.prepare("SELECT state,safe_error FROM submissions WHERE submission_id = ?").bind(first).first()).toEqual({ state: "retryable", safe_error: "operator_replay_pending" });
    expect(await env.RESULTS_DB.prepare("SELECT state FROM submissions WHERE submission_id = ?").bind(second).first<string>("state")).toBe("dead_letter");
    expect(await env.RAW_RESULTS.head(`raw/v1/2000/01/01/${first}.json`)).not.toBeNull();
  });

  test("rejects malformed percent encoding before touching D1", async () => {
    let d1Calls = 0;
    const db = proxyDatabase(() => { d1Calls += 1; });
    const response = await fetchRequest(
      new Request(`${URL}/v1/submissions/%`),
      customBindings("normal", { db }),
      context(),
    );
    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: "invalid_submission_id" });
    expect(d1Calls).toBe(0);
  });

  test("health exposes exact limits and resolved mode, never bindings or raw config", async () => {
    const workerEnv = customBindings("unknown-RAW-MODE");
    Object.assign(workerEnv, { SECRET_TOKEN: "LEAK_ME", DB_ID: "binding-123" });
    const response = await fetchRequest(new Request(`${URL}/healthz`), workerEnv, context());
    expect(response.status).toBe(200);
    const text = await response.text();
    expect(text).not.toContain("unknown-RAW-MODE");
    expect(text).not.toContain("LEAK_ME");
    expect(text).not.toContain("binding-123");
    expect(JSON.parse(text)).toEqual({
      status: "ok",
      ingest_mode: "reject",
      limits: {
        max_request_bytes: MAX_REQUEST_BYTES,
        max_results_per_request: MAX_RESULTS_PER_REQUEST,
        per_ip_requests_per_minute: PER_IP_REQUESTS_PER_MINUTE,
        global_envelopes_per_minute: GLOBAL_ENVELOPES_PER_MINUTE,
      },
      recovery: { stale_ms: RECOVERY_STALE_MS, limit: RECOVERY_LIMIT },
    });
  });

  test("does not log payloads, raw modes, failures, or binding details", async () => {
    const spies = ["debug", "error", "info", "log", "warn"].map((name) =>
      vi.spyOn(console, name as "log").mockImplementation(() => undefined));
    try {
      const rawMode = "UNKNOWN_MODE_MARKER_49d709";
      const payloadMarker = "PAYLOAD_MARKER_d150b1";
      const disabled = await postJson({ marker: payloadMarker }, customBindings(rawMode));
      expect(disabled.status).toBe(503);
      const invalid = await postJson({ schema_version: 1, results: [{ marker: payloadMarker }] }, bindings());
      expect(invalid.status).toBe(400);
      expect(await disabled.text()).not.toContain(rawMode);
      expect(await invalid.text()).not.toContain(payloadMarker);
      expect(spies.every((spy) => spy.mock.calls.length === 0)).toBe(true);
    } finally {
      for (const spy of spies) spy.mockRestore();
    }
  });
});

describe("Cloudflare binding plus load-bearing bounded D1 rate limits", () => {
  test("the real migration has one bounded row per rate scope", async () => {
    const columns = await env.RESULTS_DB.prepare("PRAGMA table_info('ingest_rate_limits')")
      .all<{ name: string; pk: number }>();
    expect(columns.results.map(({ name }) => name)).toEqual(["scope", "window_start", "count"]);
    expect(columns.results.find(({ name }) => name === "scope")?.pk).toBe(1);
  });

  test("uses the Cloudflare binding when present and returns 429 with Retry-After", async () => {
    const keys: string[] = [];
    const limited = customBindings("normal", {
      rateLimit: { limit: async ({ key }) => { keys.push(key); return { success: false }; } },
    });
    const response = await postJson(resultBatch(0), limited);
    expect(response.status).toBe(429);
    expect(response.headers.get("retry-after")).toBe("60");
    expect(await response.json()).toEqual({ error: "rate_limited" });
    expect(keys).toEqual([`ip:${TEST_IP}`]);
    expect(await rowCount()).toBe(0);
  });

  test("keeps D1 per-IP accounting active even when the binding allows", async () => {
    let bindingCalls = 0;
    const workerEnv = customBindings("store_only", {
      rateLimit: { limit: async () => { bindingCalls += 1; return { success: true }; } },
    });
    const now = vi.spyOn(Date, "now").mockReturnValue(1_800_000_000_001);
    try {
      for (let requestIndex = 0; requestIndex < PER_IP_REQUESTS_PER_MINUTE; requestIndex += 1) {
        const response = await postJson(resultBatch(requestIndex), workerEnv);
        expect(response.status).toBe(202);
    }
    const limited = await postJson(resultBatch(99), workerEnv);
    expect(limited.status).toBe(429);
    expect(limited.headers.get("retry-after")).toBe("60");
    expect(bindingCalls).toBe(PER_IP_REQUESTS_PER_MINUTE + 1);
    expect(await rowCount()).toBe(PER_IP_REQUESTS_PER_MINUTE);
    expect(await env.RESULTS_DB.prepare("SELECT COUNT(*) AS count FROM ingest_rate_limits").first<number>("count")).toBe(2);
    } finally {
      now.mockRestore();
    }

  });
  test("resets the bounded per-scope row at a new minute window", async () => {
    const currentWindow = Math.floor(Date.now() / 60_000) * 60_000;
    await env.RESULTS_DB.prepare(
      "INSERT INTO ingest_rate_limits (scope,window_start,count) VALUES (?,?,?)",
    ).bind(`ip:${TEST_IP}`, currentWindow - 60_000, PER_IP_REQUESTS_PER_MINUTE).run();
    const response = await postJson(resultBatch(0), customBindings("store_only"));
    expect(response.status).toBe(202);
    expect(await env.RESULTS_DB.prepare("SELECT window_start,count FROM ingest_rate_limits WHERE scope = ?")
      .bind(`ip:${TEST_IP}`).first()).toEqual({ window_start: currentWindow, count: 1 });
    expect(await env.RESULTS_DB.prepare("SELECT COUNT(*) AS count FROM ingest_rate_limits").first<number>("count")).toBe(2);
  });

  test("samples the rate window at each counter consumption across a slow boundary", async () => {
    const oldWindow = 1_800_000_000_000;
    const newWindow = oldWindow + 60_000;
    const enteredBinding = deferred();
    const releaseBinding = deferred();
    let bindingCalls = 0;
    const dateNow = vi.spyOn(Date, "now").mockReturnValue(oldWindow + 59_999);
    try {
      const workerEnv = customBindings("store_only", {
        rateLimit: {
          limit: async () => {
            bindingCalls += 1;
            if (bindingCalls === 1) {
              enteredBinding.resolve();
              await releaseBinding.promise;
            }
            return { success: true };
          },
        },
      });
      const slow = postJson(resultBatch(10), workerEnv);
      await enteredBinding.promise;
      dateNow.mockReturnValue(newWindow + 1);
      const fast = await postJson(resultBatch(11), workerEnv);
      releaseBinding.resolve();
      const resumed = await slow;
      expect([fast.status, resumed.status]).toEqual([202, 202]);
      expect(await env.RESULTS_DB.prepare("SELECT window_start,count FROM ingest_rate_limits WHERE scope = ?")
        .bind("global").first()).toEqual({ window_start: newWindow, count: 2 });
    } finally {
      releaseBinding.resolve();
      dateNow.mockRestore();
    }
  });

  test("rejects a stale global counter write instead of rolling the window backward", async () => {
    const oldWindow = 1_800_000_000_000;
    const newWindow = oldWindow + 60_000;
    const globalRunEntered = deferred();
    const releaseGlobalRun = deferred();
    let delayed = false;
    const db = proxyDatabase(async (sql, method, values) => {
      if (!delayed && method === "run" && sql.includes("INSERT INTO ingest_rate_limits") && values[0] === "global") {
        delayed = true;
        globalRunEntered.resolve();
        await releaseGlobalRun.promise;
      }
    });
    const dateNow = vi.spyOn(Date, "now").mockReturnValue(oldWindow + 59_999);
    try {
      const stale = postJson(resultBatch(20), customBindings("store_only", { db }));
      await globalRunEntered.promise;
      dateNow.mockReturnValue(newWindow + 1);
      const current = await postJson(resultBatch(21), customBindings("store_only"));
      releaseGlobalRun.resolve();
      const staleResponse = await stale;
      expect([staleResponse.status, current.status]).toEqual([429, 202]);
      expect(await env.RESULTS_DB.prepare("SELECT window_start,count FROM ingest_rate_limits WHERE scope = ?")
        .bind("global").first()).toEqual({ window_start: newWindow, count: 1 });
      expect(await rowCount()).toBe(1);
    } finally {
      releaseGlobalRun.resolve();
      dateNow.mockRestore();
    }
  });

  test("rejects a batch that would cross the global envelope limit atomically", async () => {
    const windowStart = Math.floor(Date.now() / 60_000) * 60_000;
    await env.RESULTS_DB.prepare(
      "INSERT INTO ingest_rate_limits (scope,window_start,count) VALUES (?,?,?)",
    ).bind("global", windowStart, GLOBAL_ENVELOPES_PER_MINUTE - 1).run();
    const response = await postJson(resultBatch(0, 1), customBindings("store_only"));
    expect(response.status).toBe(429);
    expect(response.headers.get("retry-after")).toBe("60");
    expect(await rowCount()).toBe(0);
    expect(await rawCount()).toBe(0);
    expect(await env.RESULTS_DB.prepare("SELECT count FROM ingest_rate_limits WHERE scope = ?").bind("global").first<number>("count"))
      .toBe(GLOBAL_ENVELOPES_PER_MINUTE - 1);
  });
});

describe("mode-gated scheduled recovery", () => {
  test.each(["store_only", "reject", undefined, "", "Normal", "unknown"])(
    "mode %s is a strict recovery no-op",
    async (mode) => {
      const submissionId = await seedSubmission(1, "received", "2000-01-01T00:00:00.000Z", null);
      let sends = 0;
      await scheduled(controller(), customBindings(mode, {
        queue: { send: async () => { sends += 1; } } as unknown as Queue,
      }), context());
      expect(sends).toBe(0);
      expect(await env.RESULTS_DB.prepare("SELECT state FROM submissions WHERE submission_id = ?").bind(submissionId).first<string>("state"))
        .toBe("received");
    },
  );

  test("a stale store_only row remains received until normal recovery sends the same id", async () => {
    const accepted = await postJson(resultBatch(0), customBindings("store_only"));
    const submissionId = (await accepted.json() as { receipts: Array<{ submission_id: string }> }).receipts[0].submission_id;
    await env.RESULTS_DB.prepare("UPDATE submissions SET updated_at = ? WHERE submission_id = ?")
      .bind("2000-01-01T00:00:00.000Z", submissionId).run();

    const messages: unknown[] = [];
    const queue = { send: async (message: unknown) => { messages.push(message); } } as unknown as Queue;
    await scheduled(controller(), customBindings("store_only", { queue }), context());
    expect(messages).toEqual([]);
    expect(await env.RESULTS_DB.prepare("SELECT state FROM submissions").first<string>("state")).toBe("received");

    await scheduled(controller(), customBindings("normal", { queue }), context());
    expect(messages).toEqual([{ submission_id: submissionId }]);
    expect(await env.RESULTS_DB.prepare("SELECT state, safe_error FROM submissions").first()).toEqual({ state: "queued", safe_error: null });
  });

  test("normal recovery bounds stale pages, refreshes queued rows, and advances failures fairly", async () => {
    const staleIds: string[] = [];
    for (let index = 1; index <= RECOVERY_LIMIT + 2; index += 1) {
      staleIds.push(await seedSubmission(
        index,
        index % 3 === 0 ? "queued" : index % 3 === 1 ? "received" : "retryable",
        "2000-01-01T00:00:00.000Z",
        "stale_error",
        index,
      ));
    }
    const freshId = await seedSubmission(999, "received", new Date().toISOString(), null);
    const failedId = staleIds[0];
    const refreshedQueuedId = staleIds.find((_, index) => (index + 1) % 3 === 0)!;
    const messages: string[] = [];
    const queue = {
      send: async ({ submission_id }: { submission_id: string }) => {
        messages.push(submission_id);
        if (submission_id === failedId) throw new Error("offline");
      },
    } as unknown as Queue;
    const workerEnv = customBindings("normal", { queue });
    const scheduledEvent = controller();

    await scheduled(scheduledEvent, workerEnv, context());
    expect(messages).toEqual(staleIds.slice(0, RECOVERY_LIMIT));
    const failed = await env.RESULTS_DB.prepare(
      "SELECT state,safe_error,retry_count,updated_at FROM submissions WHERE submission_id = ?",
    ).bind(failedId).first<{ state: string; safe_error: string; retry_count: number; updated_at: string }>();
    expect(failed).toMatchObject({ state: "retryable", safe_error: "queue_unavailable", retry_count: 2 });
    expect(Date.parse(failed!.updated_at)).toBeGreaterThan(Date.parse("2000-01-01T00:00:00.000Z"));
    const refreshed = await env.RESULTS_DB.prepare(
      "SELECT state,safe_error,updated_at FROM submissions WHERE submission_id = ?",
    ).bind(refreshedQueuedId).first<{ state: string; safe_error: string | null; updated_at: string }>();
    expect(refreshed).toMatchObject({ state: "queued", safe_error: null });
    expect(Date.parse(refreshed!.updated_at)).toBeGreaterThan(Date.parse("2000-01-01T00:00:00.000Z"));

    await scheduled(scheduledEvent, workerEnv, context());
    expect(messages).toEqual([...staleIds.slice(0, RECOVERY_LIMIT), ...staleIds.slice(RECOVERY_LIMIT)]);
    expect(messages).not.toContain(freshId);
  });
});

  test.each([undefined, "unknown"])("queue handler parks recoverable delivery for missing or unknown mode %s", async (mode) => {
    const accepted = await postJson(resultBatch(0), customBindings("store_only"));
    const submissionId = (await accepted.json() as { receipts: Array<{ submission_id: string }> }).receipts[0].submission_id;
    let rawReads = 0;
    let acked = 0;
    let retried = 0;
    const batch = { messages: [{ body: { submission_id: submissionId }, attempts: 1, ack: () => { acked += 1; }, retry: () => { retried += 1; } }] } as unknown as MessageBatch<unknown>;
    await queue(batch, customBindings(mode, { bucket: countedBucket(() => { rawReads += 1; }) }), context());
    expect({ acked, retried, rawReads }).toEqual({ acked: 1, retried: 0, rawReads: 0 });
    expect(await env.RESULTS_DB.prepare("SELECT state,safe_error FROM submissions WHERE submission_id = ?").bind(submissionId).first()).toEqual({ state: "retryable", safe_error: "ingest_paused" });
  });

describe("concurrency and early-reject regression gates", () => {
  test("reject mode returns before consuming the body or invoking rate infrastructure", async () => {
    let rateCalls = 0;
    const request = postRequest(JSON.stringify(resultBatch(0)));
    const response = await fetchRequest(request, customBindings(undefined, {
      rateLimit: { limit: async () => { rateCalls += 1; return { success: true }; } },
    }), context());
    expect(response.status).toBe(503);
    expect(request.bodyUsed).toBe(false);
    expect(rateCalls).toBe(0);
    expect({ rows: await rowCount(), raw: await rawCount() }).toEqual({ rows: 0, raw: 0 });
  });

  test("concurrent store_only duplicates converge on one received row and raw object", async () => {
    let sends = 0;
    const workerEnv = customBindings("store_only", {
      queue: { send: async () => { sends += 1; } } as unknown as Queue,
    });
    const firstEnvelope = validEnvelope(0);
    const sameSemanticEnvelope = {
      ...firstEnvelope,
      client_submission_id: "018f7a24-8f6b-7c8e-9d1b-2a3b4c5d6e70",
      idempotency_key: firstEnvelope.idempotency_key,
      submitted_at: "2026-07-29T11:00:00.000Z",
    } as ResultEnvelopeV1;
    const [first, second] = await Promise.all([
      postJson({ schema_version: 1, results: [firstEnvelope] }, workerEnv),
      postJson({ schema_version: 1, results: [sameSemanticEnvelope] }, workerEnv),
    ]);
    expect(first.status).toBe(202);
    expect(second.status).toBe(202);
    const firstReceipt = (await first.json() as { receipts: Array<{ submission_id: string }> }).receipts[0];
    const secondReceipt = (await second.json() as { receipts: Array<{ submission_id: string }> }).receipts[0];
    expect(firstReceipt).toEqual(secondReceipt);
    expect({ sends, rows: await rowCount(), raw: await rawCount() }).toEqual({ sends: 0, rows: 1, raw: 1 });
    expect(await env.RESULTS_DB.prepare("SELECT state FROM submissions").first<string>("state")).toBe("received");
  });

  test("bounds per-envelope storage concurrency while preserving receipt order", async () => {
    let activePuts = 0;
    let maxInflightPuts = 0;
    const bucket = proxyBucket({
      put: async (key, value, options) => {
        activePuts += 1;
        maxInflightPuts = Math.max(maxInflightPuts, activePuts);
        try {
          await new Promise<void>((resolve) => setTimeout(resolve, 5));
          return await env.RAW_RESULTS.put(key, value, options);
        } finally {
          activePuts -= 1;
        }
      },
    });
    const response = await postJson(uniqueResultBatch(100), customBindings("store_only", { bucket }));
    expect(response.status).toBe(202);
    const body = await response.json() as { receipts: Array<{ idempotency_key: string }>; errors?: unknown[] };
    expect(body.errors).toBeUndefined();
    const expectedOrder = await Promise.all(uniqueResultBatch(100).results.map(computeIdempotency));
    expect(body.receipts.map(({ idempotency_key }) => idempotency_key)).toEqual(expectedOrder);
    expect(maxInflightPuts).toBeGreaterThan(1);
    expect(maxInflightPuts).toBeLessThanOrEqual(8);
  });

  test("keeps one hundred received duplicates below one thousand internal binding calls", async () => {
    const batch = uniqueResultBatch(100, 200);
    const seeded = await postJson(batch, customBindings("store_only"));
    expect((await seeded.json() as { receipts: unknown[] }).receipts).toHaveLength(100);
    await env.RESULTS_DB.exec("DELETE FROM ingest_rate_limits");

    let bindingCalls = 0;
    const countCall = () => {
      bindingCalls += 1;
      if (bindingCalls >= 1_000) throw new Error("internal binding call budget exceeded");
    };
    const db = proxyDatabase(countCall);
    const bucket = countedBucket(countCall);
    const queue = { send: async () => { countCall(); } } as unknown as Queue;
    const response = await postJson(batch, customBindings("normal", { db, bucket, queue }));
    expect(response.status).toBe(202);
    const body = await response.json() as { receipts: unknown[]; errors?: unknown[] };
    expect(body.errors).toBeUndefined();
    expect(body.receipts).toHaveLength(100);
    expect(bindingCalls).toBe(702);
  }, 30_000);

  test("atomic D1 per-IP accounting admits exactly 30 concurrent requests", async () => {
    const responses = await Promise.all(Array.from({ length: PER_IP_REQUESTS_PER_MINUTE + 5 }, (_, index) =>
      postJson(resultBatch(index), customBindings("store_only"))));
    const statuses = responses.map(({ status }) => status);
    expect(statuses.filter((status) => status === 202)).toHaveLength(PER_IP_REQUESTS_PER_MINUTE);
    expect(statuses.filter((status) => status === 429)).toHaveLength(5);
    expect(await rowCount()).toBe(PER_IP_REQUESTS_PER_MINUTE);
    expect(await env.RESULTS_DB.prepare("SELECT count FROM ingest_rate_limits WHERE scope = ?")
      .bind(`ip:${TEST_IP}`).first<number>("count")).toBe(PER_IP_REQUESTS_PER_MINUTE);
  });

  test("atomic global accounting admits only one concurrent final envelope", async () => {
    const windowStart = Math.floor(Date.now() / 60_000) * 60_000;
    await env.RESULTS_DB.prepare(
      "INSERT INTO ingest_rate_limits (scope,window_start,count) VALUES (?,?,?)",
    ).bind("global", windowStart, GLOBAL_ENVELOPES_PER_MINUTE - 1).run();
    const responses = await Promise.all([
      postJson(resultBatch(0), customBindings("store_only")),
      postJson(resultBatch(1), customBindings("store_only")),
    ]);
    expect(responses.map(({ status }) => status).sort()).toEqual([202, 429]);
    expect(await rowCount()).toBe(1);
    expect(await env.RESULTS_DB.prepare("SELECT count FROM ingest_rate_limits WHERE scope = ?")
      .bind("global").first<number>("count")).toBe(GLOBAL_ENVELOPES_PER_MINUTE);
  });
});
