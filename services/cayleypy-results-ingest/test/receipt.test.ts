import { env } from "cloudflare:test";
import { beforeEach, describe, expect, test } from "vitest";

import { canonicalJson, computeIdempotency } from "../src/ids.js";
import { transition } from "../src/db.js";
import { parkPausedSubmission, receiveEnvelope, recoverStaleSubmissions, SafeIngestError, type IngestEnv } from "../src/storage.js";

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

function dbWithTransitionNoop(): D1Database {
  return {
    prepare(query: string) {
      if (query.startsWith("UPDATE submissions SET")) {
        return {
          bind: () => ({
            run: async () => ({ success: true, meta: { changes: 0 } }),
          }),
        } as unknown as D1PreparedStatement;
      }
      return env.RESULTS_DB.prepare(query);
    },
  } as unknown as D1Database;
}


async function seedSubmission(
  submissionId: string,
  idempotencyKey: string,
  state: "received" | "queued" | "retryable",
  options: {
    safeError?: string | null;
    retryCount?: number;
    updatedAt?: string;
  } = {},
): Promise<string> {
  const envelope = validEnvelope();
  const rawKey = `raw/v1/2000/01/01/${submissionId}.json`;
  const updatedAt = options.updatedAt ?? "2000-01-01T00:00:00.000Z";
  await env.RAW_RESULTS.put(rawKey, canonicalJson(envelope));
  await env.RESULTS_DB.prepare(
    "INSERT INTO submissions (submission_id,idempotency_key,run_id,author_name,competition,puzzle_type,puzzle_id,state,raw_r2_key,safe_error,retry_count,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
  ).bind(
    submissionId,
    idempotencyKey,
    `${envelope.run_id}-${submissionId}`,
    envelope.author.name,
    envelope.competition,
    envelope.puzzle_type,
    envelope.puzzle_id,
    state,
    rawKey,
    options.safeError ?? null,
    options.retryCount ?? 0,
    "2000-01-01T00:00:00.000Z",
    updatedAt,
  ).run();
  return rawKey;
}

beforeEach(async () => {
  await env.RESULTS_DB.exec("DELETE FROM submissions");
  const listed = await env.RAW_RESULTS.list({ prefix: "raw/" });
  if (listed.objects.length) await env.RAW_RESULTS.delete(listed.objects.map((object: R2Object) => object.key));
});

describe("receipt durability with Miniflare D1 and R2 bindings", () => {
  test("applies the deployable migration, state CHECK, and recovery index", async () => {
    const appliedMigration = await env.RESULTS_DB.prepare("SELECT name FROM d1_migrations").first<string>("name");
    expect(appliedMigration).toBe("0001_initial.sql");

    const index = await env.RESULTS_DB.prepare("PRAGMA index_info('submissions_recovery')")
      .all<{ seqno: number; cid: number; name: string }>();
    expect(index.results.map((column) => column.name)).toEqual(["state", "updated_at"]);

    const envelope = validEnvelope();
    const invalidInsert = env.RESULTS_DB.prepare(
      "INSERT INTO submissions (submission_id,idempotency_key,run_id,author_name,competition,puzzle_type,puzzle_id,state,raw_r2_key,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
    ).bind(
      "019c9999-0000-7000-8000-000000000000",
      "f".repeat(64),
      envelope.run_id,
      envelope.author.name,
      envelope.competition,
      envelope.puzzle_type,
      envelope.puzzle_id,
      "terminal",
      "raw/v1/invalid-state.json",
      "2000-01-01T00:00:00.000Z",
      "2000-01-01T00:00:00.000Z",
    ).run();
    await expect(invalidInsert).rejects.toThrow();
    expect(await env.RESULTS_DB.prepare("SELECT COUNT(*) AS count FROM submissions").first<number>("count")).toBe(0);
  });
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
  test("resends a stale received duplicate after the bounded wait", async () => {
    const queueGate = deferred();
    const firstQueueStarted = deferred();
    const secondQueueStarted = deferred();
    let enqueueCount = 0;
    const bindings: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: {
        send: async () => {
          enqueueCount += 1;
          if (enqueueCount === 1) firstQueueStarted.resolve();
          if (enqueueCount === 2) secondQueueStarted.resolve();
          await queueGate.promise;
        },
      } as unknown as Queue,
    };
    const winnerPending = receiveEnvelope(bindings, validEnvelope());
    await firstQueueStarted.promise;
    const duplicatePending = receiveEnvelope(bindings, validEnvelope(), {
      duplicatePollAttempts: 1,
      duplicatePollDelayMs: 0,
    });
    await secondQueueStarted.promise;
    queueGate.resolve();
    const [winner, duplicate] = await Promise.all([winnerPending, duplicatePending]);
    expect(duplicate).toMatchObject({ submission_id: winner.submission_id, state: "queued", duplicate: true });
    expect(enqueueCount).toBe(2);
    expect(await env.RESULTS_DB.prepare("SELECT COUNT(*) AS count FROM submissions").first<number>("count")).toBe(1);
  });

  test("scheduled recovery repairs a crash-before-send received row with the same service message", async () => {
    const envelope = validEnvelope();
    const idempotency = await computeIdempotency(envelope);
    const submissionId = "019c1234-5678-7abc-8def-0123456789ab";
    const rawKey = `raw/v1/2000/01/01/${submissionId}.json`;
    await env.RAW_RESULTS.put(rawKey, canonicalJson(envelope));
    await env.RESULTS_DB.prepare(
      "INSERT INTO submissions (submission_id,idempotency_key,run_id,author_name,competition,puzzle_type,puzzle_id,state,raw_r2_key,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
    ).bind(
      submissionId,
      idempotency,
      envelope.run_id,
      envelope.author.name,
      envelope.competition,
      envelope.puzzle_type,
      envelope.puzzle_id,
      "received",
      rawKey,
      "2000-01-01T00:00:00.000Z",
      "2000-01-01T00:00:00.000Z",
    ).run();
    const messages: unknown[] = [];
    const recovery = await recoverStaleSubmissions(
      {
        RESULTS_DB: env.RESULTS_DB,
        RAW_RESULTS: env.RAW_RESULTS,
        VALIDATE_QUEUE: { send: async (message: unknown) => { messages.push(message); } } as unknown as Queue,
      },
      { staleBefore: new Date("2001-01-01T00:00:00.000Z"), limit: 10 },
    );
    expect(recovery).toEqual({ scanned: 1, queued: 1, retryable: 0, failed: 0 });
    expect(messages).toEqual([{ submission_id: submissionId }]);
    expect(await env.RESULTS_DB.prepare("SELECT state FROM submissions").first<string>("state")).toBe("queued");
    expect(await env.RAW_RESULTS.head(rawKey)).not.toBeNull();
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
    const row = await env.RESULTS_DB.prepare("SELECT state, safe_error, retry_count, raw_r2_key FROM submissions WHERE submission_id = ?").bind(receipt.submission_id).first<{ state: string; safe_error: string; retry_count: number; raw_r2_key: string }>();
    expect(receipt.state).toBe("retryable");
    expect(row).toMatchObject({ state: "retryable", safe_error: "queue_unavailable", retry_count: 1 });
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

  test("successful Queue send advances a concurrent retryable row and clears its stale error", async () => {
    const bindings: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: {
        send: async ({ submission_id }: { submission_id: string }) => {
          await env.RESULTS_DB.prepare("UPDATE submissions SET state = ?, safe_error = ? WHERE submission_id = ?")
            .bind("retryable", "queue_unavailable", submission_id)
            .run();
        },
      } as unknown as Queue,
    };
    await expect(receiveEnvelope(bindings, validEnvelope())).resolves.toMatchObject({ state: "queued" });
    const row = await env.RESULTS_DB.prepare("SELECT state, safe_error FROM submissions").first<{ state: string; safe_error: string | null }>();
    expect(row).toEqual({ state: "queued", safe_error: null });
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

  test("returns queued when ambiguous Queue failure already reached the consumer", async () => {
    const bindings: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: {
        send: async ({ submission_id }: { submission_id: string }) => {
          await env.RESULTS_DB.prepare("UPDATE submissions SET state = ? WHERE submission_id = ?")
            .bind("validating", submission_id)
            .run();
          throw new Error("ambiguous_queue_failure");
        },
      } as unknown as Queue,
    };
    await expect(receiveEnvelope(bindings, validEnvelope())).resolves.toMatchObject({ state: "queued" });
  });

  test("scheduled recovery re-enqueues stale retryable rows and retains raw", async () => {
    const failing: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: { send: async () => { throw new Error("offline"); } } as unknown as Queue,
    };
    const receipt = await receiveEnvelope(failing, validEnvelope());
    await env.RESULTS_DB.prepare("UPDATE submissions SET updated_at = ? WHERE submission_id = ?")
      .bind("2000-01-01T00:00:00.000Z", receipt.submission_id)
      .run();
    const messages: unknown[] = [];
    const recovery = await recoverStaleSubmissions(
      {
        RESULTS_DB: env.RESULTS_DB,
        RAW_RESULTS: env.RAW_RESULTS,
        VALIDATE_QUEUE: { send: async (message: unknown) => { messages.push(message); } } as unknown as Queue,
      },
      { staleBefore: new Date("2001-01-01T00:00:00.000Z"), limit: 10 },
    );
    expect(recovery).toEqual({ scanned: 1, queued: 1, retryable: 0, failed: 0 });
    expect(messages).toEqual([{ submission_id: receipt.submission_id }]);
    const row = await env.RESULTS_DB.prepare(
      "SELECT state, safe_error, retry_count, raw_r2_key FROM submissions",
    ).first<{ state: string; safe_error: string | null; retry_count: number; raw_r2_key: string }>();
    expect(row).toMatchObject({ state: "queued", safe_error: null, retry_count: 1 });
    expect(await env.RAW_RESULTS.head(row!.raw_r2_key)).not.toBeNull();
  });

  test("failed recovery advances retry metadata so a bounded page cannot starve its tail", async () => {
    const envelope = validEnvelope();
    const submissionIds: string[] = [];
    for (let index = 0; index < 3; index += 1) {
      const submissionId = `019c0000-0000-7000-8000-00000000000${index}`;
      const rawKey = `raw/v1/2000/01/01/${submissionId}.json`;
      submissionIds.push(submissionId);
      await env.RAW_RESULTS.put(rawKey, canonicalJson(envelope));
      await env.RESULTS_DB.prepare(
        "INSERT INTO submissions (submission_id,idempotency_key,run_id,author_name,competition,puzzle_type,puzzle_id,state,raw_r2_key,safe_error,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
      ).bind(
        submissionId,
        (index + 1).toString(16).padStart(64, "0"),
        `${envelope.run_id}-${index}`,
        envelope.author.name,
        envelope.competition,
        envelope.puzzle_type,
        envelope.puzzle_id + index,
        "retryable",
        rawKey,
        "queue_unavailable",
        "2000-01-01T00:00:00.000Z",
        "2000-01-01T00:00:00.000Z",
      ).run();
    }

    const messages: string[] = [];
    const recoveryEnv: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: {
        send: async ({ submission_id }: { submission_id: string }) => {
          messages.push(submission_id);
          throw new Error("still_offline");
        },
      } as unknown as Queue,
    };
    const options = { staleBefore: new Date("2001-01-01T00:00:00.000Z"), limit: 2 };

    await expect(recoverStaleSubmissions(recoveryEnv, options)).resolves.toEqual({ scanned: 2, queued: 0, retryable: 2, failed: 0 });
    await expect(recoverStaleSubmissions(recoveryEnv, options)).resolves.toEqual({ scanned: 1, queued: 0, retryable: 1, failed: 0 });
    await expect(recoverStaleSubmissions(recoveryEnv, options)).resolves.toEqual({ scanned: 0, queued: 0, retryable: 0, failed: 0 });
    expect(messages).toEqual(submissionIds);
    const rows = await env.RESULTS_DB.prepare("SELECT submission_id, retry_count FROM submissions ORDER BY submission_id").all<{ submission_id: string; retry_count: number }>();
    expect(rows.results).toEqual(submissionIds.map((submission_id) => ({ submission_id, retry_count: 1 })));
  });

  test("parks received, queued, and retryable rows without consuming retry budget or raw", async () => {
    const fixtures = [
      {
        submissionId: "019c1000-0000-7000-8000-000000000001",
        idempotencyKey: "1".repeat(64),
        state: "received",
        retryCount: 2,
      },
      {
        submissionId: "019c1000-0000-7000-8000-000000000002",
        idempotencyKey: "2".repeat(64),
        state: "queued",
        retryCount: 3,
      },
      {
        submissionId: "019c1000-0000-7000-8000-000000000003",
        idempotencyKey: "3".repeat(64),
        state: "retryable",
        retryCount: 4,
      },
    ] as const;
    const rawKeys = new Map<string, string>();
    for (const fixture of fixtures) {
      rawKeys.set(
        fixture.submissionId,
        await seedSubmission(
          fixture.submissionId,
          fixture.idempotencyKey,
          fixture.state,
          { safeError: "stale_error", retryCount: fixture.retryCount },
        ),
      );
    }

    const park = parkPausedSubmission;
    for (const fixture of fixtures) {
      const row = await park(env.RESULTS_DB, fixture.submissionId);
      expect(row).toMatchObject({
        submission_id: fixture.submissionId,
        state: "retryable",
        safe_error: "ingest_paused",
        retry_count: fixture.retryCount,
        raw_r2_key: rawKeys.get(fixture.submissionId),
      });
      expect(Date.parse(row.updated_at)).toBeGreaterThan(Date.parse("2000-01-01T00:00:00.000Z"));
      expect(await env.RAW_RESULTS.head(row.raw_r2_key)).not.toBeNull();
    }
  });

  test("duplicate pause deliveries beyond a max-retries-equivalent remain one parked row", async () => {
    const submissionId = "019c1000-0000-7000-8000-000000000004";
    const rawKey = await seedSubmission(
      submissionId,
      "4".repeat(64),
      "retryable",
      { safeError: "ingest_paused", retryCount: 7 },
    );
    const park = parkPausedSubmission;

    for (let delivery = 0; delivery < 101; delivery += 1) {
      await expect(park(env.RESULTS_DB, submissionId)).resolves.toMatchObject({
        submission_id: submissionId,
        state: "retryable",
        safe_error: "ingest_paused",
        retry_count: 7,
      });
    }

    expect(await env.RESULTS_DB.prepare("SELECT COUNT(*) AS count FROM submissions").first<number>("count")).toBe(1);
    expect(await env.RAW_RESULTS.head(rawKey)).not.toBeNull();
  });

  test("accepts a false pause transition only after reread verifies the durable park", async () => {
    const submissionId = "019c1000-0000-7000-8000-000000000005";
    const rawKey = await seedSubmission(
      submissionId,
      "5".repeat(64),
      "retryable",
      { safeError: "ingest_paused", retryCount: 8 },
    );

    await expect(parkPausedSubmission(dbWithTransitionNoop(), submissionId)).resolves.toEqual({
      submission_id: submissionId,
      idempotency_key: "5".repeat(64),
      state: "retryable",
      raw_r2_key: rawKey,
      safe_error: "ingest_paused",
      retry_count: 8,
      updated_at: "2000-01-01T00:00:00.000Z",
    });
  });

  test("normal recovery rescues one stale queued row after repeated D1 park failures", async () => {
    const submissionId = "019c1000-0000-7000-8000-000000000006";
    const rawKey = await seedSubmission(
      submissionId,
      "6".repeat(64),
      "queued",
      { safeError: "stale_error", retryCount: 9 },
    );
    const park = parkPausedSubmission;
    for (let attempt = 0; attempt < 4; attempt += 1) {
      await expect(park(dbWithTransitionFailure(), submissionId)).rejects.toMatchObject({
        code: "state_transition_failed",
      });
    }

    const messages: unknown[] = [];
    const recoveryEnv: IngestEnv = {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: { send: async (message: unknown) => { messages.push(message); } } as unknown as Queue,
    };
    await expect(recoverStaleSubmissions(
      recoveryEnv,
      { staleBefore: new Date("2001-01-01T00:00:00.000Z"), limit: 10 },
    )).resolves.toEqual({ scanned: 1, queued: 1, retryable: 0, failed: 0 });
    expect(messages).toEqual([{ submission_id: submissionId }]);
    const row = await env.RESULTS_DB.prepare(
      "SELECT state, safe_error, retry_count, raw_r2_key, updated_at FROM submissions",
    ).first<{ state: string; safe_error: string | null; retry_count: number; raw_r2_key: string; updated_at: string }>();
    expect(row).toMatchObject({ state: "queued", safe_error: null, retry_count: 9, raw_r2_key: rawKey });
    expect(Date.parse(row!.updated_at)).toBeGreaterThan(Date.parse("2000-01-01T00:00:00.000Z"));
    expect(await env.RAW_RESULTS.head(rawKey)).not.toBeNull();
    await expect(recoverStaleSubmissions(
      recoveryEnv,
      { staleBefore: new Date("2001-01-01T00:00:00.000Z"), limit: 10 },
    )).resolves.toEqual({ scanned: 0, queued: 0, retryable: 0, failed: 0 });
  });

  test("compare-and-transition changes exactly one eligible row", async () => {
    const bindings: IngestEnv = { RESULTS_DB: env.RESULTS_DB, RAW_RESULTS: env.RAW_RESULTS, VALIDATE_QUEUE: { send: async () => undefined } as unknown as Queue };
    const receipt = await receiveEnvelope(bindings, validEnvelope());
    await expect(transition(env.RESULTS_DB, receipt.submission_id, ["queued"], "validating")).resolves.toBe(true);
    await expect(transition(env.RESULTS_DB, receipt.submission_id, ["queued"], "validated")).resolves.toBe(false);
  });
});
