import { env } from "cloudflare:test";
import { beforeEach, describe, expect, test } from "vitest";

import { canonicalJson, computeIdempotency } from "../src/ids.js";
import { transition } from "../src/db.js";
import { receiveEnvelope, SafeIngestError, type IngestEnv } from "../src/storage.js";

declare module "cloudflare:test" { interface ProvidedEnv extends Pick<IngestEnv, "RESULTS_DB" | "RAW_RESULTS"> {} }

const schema = "CREATE TABLE IF NOT EXISTS submissions (submission_id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE, run_id TEXT NOT NULL, author_name TEXT NOT NULL, competition TEXT NOT NULL, puzzle_type TEXT NOT NULL, puzzle_id INTEGER NOT NULL, state TEXT NOT NULL, raw_r2_key TEXT NOT NULL UNIQUE, safe_error TEXT, retry_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, github_path TEXT, github_commit_sha TEXT)";

const validEnvelope = () => ({
  schema_version: 1 as const, submission_id: "018f7a24-8f6b-7c8e-9d1b-2a3b4c5d6e7f", run_id: "run-20260728-001", idempotency_key: "a".repeat(64),
  author: { name: "Ada", verification: "claimed" as const }, kaggle: { owner: "ada", slug: "run", version: 1 }, competition: "santa-2023", puzzle_type: "cube", puzzle_id: 42,
  proof: { initial_state: [0, 1, 2], central_state: [1, 2, 0], generators: { r: [1, 2, 0] } },
  orientation: { search_mode: "off" as const, final_orientation: "original" as const }, solution: { path: ["r"], length: 1, solved_depth: 1, validation: "valid" as const },
  profile: { requested_beam: 1024, effective_beam: 1024, alignment_delta: 0, evidence: "t4-v1" }, runtime: { touch_bfs_radius: 1, solution_mode: "first" as const, max_depth: 10, max_collected_solutions: 1 },
  model: { filename: "model.pth", sha256: "b".repeat(64), format: "batchnorm-folded" as const, manifest: { output_dim: 1 } }, hardware: { gpu_names: ["Tesla T4"], world_size: 2, total_runtime_ms: 1 }, solver_commit: "c".repeat(40), submitted_at: "2026-07-28T10:00:00.000Z",
});

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve: (() => void) | undefined;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve: () => resolve!() };
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

function dbWithInsertFailure(): D1Database {
  return {
    prepare(query: string) {
      if (query.startsWith("INSERT INTO submissions")) {
        return {
          bind: () => ({ run: async () => { throw new Error("injected_d1_insert_failure"); } }),
        } as unknown as D1PreparedStatement;
      }
      return env.RESULTS_DB.prepare(query);
    },
  } as unknown as D1Database;
}

function dbReleasingQueueOnLastIdempotencyRead(releaseQueue: () => void): D1Database {
  let idempotencyReads = 0;
  return {
    prepare(query: string) {
      const statement = env.RESULTS_DB.prepare(query);
      if (!query.includes("FROM submissions WHERE idempotency_key = ?")) return statement;
      return {
        bind(...values: unknown[]) {
          const bound = statement.bind(...values);
          return {
            async first<T>() {
              idempotencyReads += 1;
              if (idempotencyReads === 2) {
                releaseQueue();
                for (let attempt = 0; attempt < 100; attempt += 1) {
                  const state = await env.RESULTS_DB.prepare("SELECT state FROM submissions").first<string>("state");
                  if (state !== "received") break;
                  await Promise.resolve();
                }
              }
              return bound.first<T>();
            },
          } as unknown as D1PreparedStatement;
        },
      } as unknown as D1PreparedStatement;
    },
  } as unknown as D1Database;
}
function dbWithTransitionFailure(): D1Database {
  return {
    prepare(query: string) {
      if (query.startsWith("UPDATE submissions SET")) {
        return {
          bind: () => ({ run: async () => { throw new Error("injected_d1_transition_failure"); } }),
        } as unknown as D1PreparedStatement;
      }
      return env.RESULTS_DB.prepare(query);
    },
  } as unknown as D1Database;
}

beforeEach(async () => {
  await env.RESULTS_DB.exec(schema);
  await env.RESULTS_DB.exec("DELETE FROM submissions");
  const listed = await env.RAW_RESULTS.list({ prefix: "raw/" });
  if (listed.objects.length) await env.RAW_RESULTS.delete(listed.objects.map((object: R2Object) => object.key));
});

describe("receipt durability with Miniflare D1 and R2 bindings", () => {
  test("canonical JSON is key-order stable and semantic idempotency excludes transport ids", async () => {
    expect(canonicalJson({ z: [true, null], a: 1 })).toBe('{"a":1,"z":[true,null]}');
    const changedTransport = { ...validEnvelope(), submission_id: "018f7a24-8f6b-7c8e-9d1b-2a3b4c5d6e70", idempotency_key: "d".repeat(64), submitted_at: "2026-07-28T11:00:00.000Z" };
    await expect(computeIdempotency(validEnvelope())).resolves.toBe(await computeIdempotency(changedTransport));
  });

  test("orders R2 durability, D1 received state, Queue durability, and receipt", async () => {
    const r2Gate = deferred();
    const queueGate = deferred();
    const queueStarted = deferred();
    const rawBucket = proxyBucket({
      put: async (...args: Parameters<R2Bucket["put"]>) => { await r2Gate.promise; return env.RAW_RESULTS.put(...args); },
    });
    const queue = { send: async () => { queueStarted.resolve(); await queueGate.promise; } } as unknown as Queue;
    const bindings: IngestEnv = { RESULTS_DB: env.RESULTS_DB, RAW_RESULTS: rawBucket, VALIDATE_QUEUE: queue };
    let settled = false;
    const pending = receiveEnvelope(bindings, validEnvelope()).then((receipt) => { settled = true; return receipt; });
    await Promise.resolve();
    expect(await env.RESULTS_DB.prepare("SELECT submission_id FROM submissions").first()).toBeNull();
    expect(settled).toBe(false);
    r2Gate.resolve();
    await queueStarted.promise;
    expect((await env.RESULTS_DB.prepare("SELECT state FROM submissions").first<{ state: string }>())?.state).toBe("received");
    expect(settled).toBe(false);
    queueGate.resolve();
    await expect(pending).resolves.toMatchObject({ state: "queued", duplicate: false });
  });

  test("holds a concurrent duplicate until the winner Queue write is durable", async () => {
    const queueGate = deferred();
    const queueStarted = deferred();
    let enqueueCount = 0;
    const bindings: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: {
        send: async () => { enqueueCount += 1; queueStarted.resolve(); await queueGate.promise; },
      } as unknown as Queue,
    };
    const winnerPending = receiveEnvelope(bindings, validEnvelope());
    await queueStarted.promise;
    let duplicateSettled = false;
    const duplicatePending = receiveEnvelope(
      bindings,
      { ...validEnvelope(), submission_id: "018f7a24-8f6b-7c8e-9d1b-2a3b4c5d6e70" },
      { duplicatePollAttempts: 100, duplicatePollDelayMs: 1 },
    ).then((receipt) => { duplicateSettled = true; return receipt; });
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(duplicateSettled).toBe(false);
    expect(enqueueCount).toBe(1);
    queueGate.resolve();
    const [winner, duplicate] = await Promise.all([winnerPending, duplicatePending]);
    expect(duplicate).toMatchObject({ submission_id: winner.submission_id, state: "queued", duplicate: true });
    expect(enqueueCount).toBe(1);
  });

  test("accepts a winner that settles on the final bounded reread", async () => {
    const queueGate = deferred();
    const queueStarted = deferred();
    const winnerBindings: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: { send: async () => { queueStarted.resolve(); await queueGate.promise; } } as unknown as Queue,
    };
    const winnerPending = receiveEnvelope(winnerBindings, validEnvelope());
    await queueStarted.promise;
    const duplicateBindings: IngestEnv = {
      ...winnerBindings,
      RESULTS_DB: dbReleasingQueueOnLastIdempotencyRead(queueGate.resolve),
    };
    const duplicate = await receiveEnvelope(duplicateBindings, validEnvelope(), {
      duplicatePollAttempts: 1,
      duplicatePollDelayMs: 0,
    });
    const winner = await winnerPending;
    expect(duplicate).toMatchObject({ submission_id: winner.submission_id, state: "queued", duplicate: true });
  });
  test("fails safely when a received duplicate does not settle before the bound", async () => {
    const queueGate = deferred();
    const queueStarted = deferred();
    const bindings: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: { send: async () => { queueStarted.resolve(); await queueGate.promise; } } as unknown as Queue,
    };
    const winnerPending = receiveEnvelope(bindings, validEnvelope());
    await queueStarted.promise;
    await expect(receiveEnvelope(bindings, validEnvelope(), {
      duplicatePollAttempts: 2,
      duplicatePollDelayMs: 0,
    })).rejects.toMatchObject({ code: "duplicate_wait_timeout" });
    queueGate.resolve();
    await winnerPending;
  });

  test("retains immutable raw when the D1 insert outcome is ambiguous", async () => {
    const bindings: IngestEnv = {
      RESULTS_DB: dbWithInsertFailure(),
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: { send: async () => { throw new Error("queue_must_not_run"); } } as unknown as Queue,
    };
    await expect(receiveEnvelope(bindings, validEnvelope())).rejects.toMatchObject({ code: "submission_persist_failed" });
    const raw = await env.RAW_RESULTS.list({ prefix: "raw/" });
    expect(raw.objects).toHaveLength(1);
    expect(await env.RAW_RESULTS.head(raw.objects[0].key)).not.toBeNull();
    expect(await env.RESULTS_DB.prepare("SELECT submission_id FROM submissions").first()).toBeNull();
  });
  test("removes only the D1 conflict loser service-generated raw object", async () => {
    const bothPutsStarted = deferred();
    const releasePuts = deferred();
    let putCount = 0;
    let enqueueCount = 0;
    const bucket = proxyBucket({
      put: async (...args: Parameters<R2Bucket["put"]>) => {
        putCount += 1;
        if (putCount === 2) bothPutsStarted.resolve();
        await releasePuts.promise;
        return env.RAW_RESULTS.put(...args);
      },
    });
    const bindings: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: bucket,
      VALIDATE_QUEUE: { send: async () => { enqueueCount += 1; } } as unknown as Queue,
    };
    const first = receiveEnvelope(bindings, validEnvelope());
    const second = receiveEnvelope(bindings, {
      ...validEnvelope(),
      submission_id: "018f7a24-8f6b-7c8e-9d1b-2a3b4c5d6e70",
    });
    await bothPutsStarted.promise;
    releasePuts.resolve();
    const receipts = await Promise.all([first, second]);
    expect(receipts[0].submission_id).toBe(receipts[1].submission_id);
    expect(enqueueCount).toBe(1);
    const winner = await env.RESULTS_DB.prepare("SELECT raw_r2_key FROM submissions").first<{ raw_r2_key: string }>();
    const raw = await env.RAW_RESULTS.list({ prefix: "raw/" });
    expect(raw.objects.map((object) => object.key)).toEqual([winner!.raw_r2_key]);
    expect(await env.RAW_RESULTS.head(winner!.raw_r2_key)).not.toBeNull();
  });

  test("does not claim duplicate success when loser raw cleanup is unconfirmed", async () => {
    const bothPutsStarted = deferred();
    const releasePuts = deferred();
    let putCount = 0;
    const bucket = proxyBucket({
      put: async (...args: Parameters<R2Bucket["put"]>) => {
        putCount += 1;
        if (putCount === 2) bothPutsStarted.resolve();
        await releasePuts.promise;
        return env.RAW_RESULTS.put(...args);
      },
      delete: async () => undefined,
    });
    const bindings: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: bucket,
      VALIDATE_QUEUE: { send: async () => undefined } as unknown as Queue,
    };
    const first = receiveEnvelope(bindings, validEnvelope());
    const second = receiveEnvelope(bindings, {
      ...validEnvelope(),
      submission_id: "018f7a24-8f6b-7c8e-9d1b-2a3b4c5d6e70",
    });
    await bothPutsStarted.promise;
    releasePuts.resolve();
    const settled = await Promise.allSettled([first, second]);
    expect(settled.filter((result) => result.status === "fulfilled")).toHaveLength(1);
    const rejected = settled.find((result): result is PromiseRejectedResult => result.status === "rejected");
    expect(rejected?.reason).toBeInstanceOf(SafeIngestError);
    expect(rejected?.reason).toMatchObject({ code: "duplicate_raw_cleanup_failed" });
  });

  test("writes immutable R2 raw JSON before the D1 receipt and enqueues once", async () => {
    const messages: unknown[] = [];
    const bindings: IngestEnv = { RESULTS_DB: env.RESULTS_DB, RAW_RESULTS: env.RAW_RESULTS, VALIDATE_QUEUE: { send: async (message: unknown) => { messages.push(message); } } as unknown as Queue };
    const first = await receiveEnvelope(bindings, validEnvelope(), { receivedAt: new Date("2026-07-28T12:00:00Z") });
    const row = await env.RESULTS_DB.prepare("SELECT state, raw_r2_key FROM submissions WHERE submission_id = ?").bind(first.submission_id).first<{ state: string; raw_r2_key: string }>();
    expect(first).toMatchObject({ state: "queued", duplicate: false });
    expect(messages).toEqual([{ submission_id: first.submission_id }]);
    expect(row?.state).toBe("queued");
    const raw = await env.RAW_RESULTS.get(row!.raw_r2_key);
    expect(raw?.customMetadata?.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(await raw?.text()).toBe(canonicalJson(validEnvelope()));

    const duplicate = await receiveEnvelope(bindings, { ...validEnvelope(), submission_id: "018f7a24-8f6b-7c8e-9d1b-2a3b4c5d6e70" });
    expect(duplicate).toMatchObject({ submission_id: first.submission_id, duplicate: true });
    expect(messages).toHaveLength(1);
  });

  test("retains R2 object and a retryable row when durable queue write fails", async () => {
    const bindings: IngestEnv = { RESULTS_DB: env.RESULTS_DB, RAW_RESULTS: env.RAW_RESULTS, VALIDATE_QUEUE: { send: async () => { throw new Error("injected"); } } as unknown as Queue };
    const receipt = await receiveEnvelope(bindings, validEnvelope());
    const row = await env.RESULTS_DB.prepare("SELECT state, safe_error, raw_r2_key FROM submissions WHERE submission_id = ?").bind(receipt.submission_id).first<{ state: string; safe_error: string; raw_r2_key: string }>();
    expect(receipt.state).toBe("retryable");
    expect(row).toMatchObject({ state: "retryable", safe_error: "queue_unavailable" });
    expect(await env.RAW_RESULTS.head(row!.raw_r2_key)).not.toBeNull();
  });

  test("accepts transition false only when reread proves a queued-or-later state", async () => {
    const bindings: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: {
        send: async ({ submission_id }: { submission_id: string }) => {
          await env.RESULTS_DB.prepare("UPDATE submissions SET state = ? WHERE submission_id = ?").bind("validating", submission_id).run();
        },
      } as unknown as Queue,
    };
    await expect(receiveEnvelope(bindings, validEnvelope())).resolves.toMatchObject({ state: "queued", duplicate: false });
  });

  test("fails safely when successful Queue send conflicts with a retryable state", async () => {
    const bindings: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: {
        send: async ({ submission_id }: { submission_id: string }) => {
          await env.RESULTS_DB.prepare("UPDATE submissions SET state = ? WHERE submission_id = ?").bind("retryable", submission_id).run();
        },
      } as unknown as Queue,
    };
    await expect(receiveEnvelope(bindings, validEnvelope())).rejects.toMatchObject({ code: "state_transition_conflict" });
  });

  test("does not mislabel a D1 transition error as Queue unavailable", async () => {
    const bindings: IngestEnv = {
      RESULTS_DB: dbWithTransitionFailure(),
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: { send: async () => { throw new Error("injected_queue_failure"); } } as unknown as Queue,
    };
    await expect(receiveEnvelope(bindings, validEnvelope())).rejects.toMatchObject({ code: "state_transition_failed" });
    const row = await env.RESULTS_DB.prepare("SELECT state, safe_error FROM submissions").first<{ state: string; safe_error: string | null }>();
    expect(row).toEqual({ state: "received", safe_error: null });
  });

  test("does not return retryable when Queue failure races with another state", async () => {
    const bindings: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: {
        send: async ({ submission_id }: { submission_id: string }) => {
          await env.RESULTS_DB.prepare("UPDATE submissions SET state = ? WHERE submission_id = ?").bind("queued", submission_id).run();
          throw new Error("injected_queue_failure");
        },
      } as unknown as Queue,
    };
    await expect(receiveEnvelope(bindings, validEnvelope())).rejects.toMatchObject({ code: "state_transition_conflict" });
  });

  test("compare-and-transition changes exactly one eligible row", async () => {
    const bindings: IngestEnv = { RESULTS_DB: env.RESULTS_DB, RAW_RESULTS: env.RAW_RESULTS, VALIDATE_QUEUE: { send: async () => undefined } as unknown as Queue };
    const receipt = await receiveEnvelope(bindings, validEnvelope());
    await expect(transition(env.RESULTS_DB, receipt.submission_id, ["queued"], "validating")).resolves.toBe(true);
    await expect(transition(env.RESULTS_DB, receipt.submission_id, ["queued"], "validated")).resolves.toBe(false);
  });
});
