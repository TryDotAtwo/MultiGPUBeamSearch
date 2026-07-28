import { env } from "cloudflare:test";
import { beforeEach, describe, expect, test } from "vitest";

import { canonicalJson, computeIdempotency } from "../src/ids.js";
import { transition } from "../src/db.js";
import { receiveEnvelope, type IngestEnv } from "../src/storage.js";

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
    let releaseR2: (() => void) | undefined;
    let releaseQueue: (() => void) | undefined;
    let queueStarted: (() => void) | undefined;
    const r2Gate = new Promise<void>((resolve) => { releaseR2 = resolve; });
    const queueGate = new Promise<void>((resolve) => { releaseQueue = resolve; });
    const queueStartedGate = new Promise<void>((resolve) => { queueStarted = resolve; });
    const rawBucket = {
      ...env.RAW_RESULTS,
      put: async (...args: Parameters<R2Bucket["put"]>) => { await r2Gate; return env.RAW_RESULTS.put(...args); },
    } as unknown as R2Bucket;
    const queue = { send: async () => { queueStarted!(); await queueGate; } } as unknown as Queue;
    const bindings: IngestEnv = { RESULTS_DB: env.RESULTS_DB, RAW_RESULTS: rawBucket, VALIDATE_QUEUE: queue };
    let settled = false;
    const pending = receiveEnvelope(bindings, validEnvelope()).then((receipt) => { settled = true; return receipt; });
    await Promise.resolve();
    expect(await env.RESULTS_DB.prepare("SELECT submission_id FROM submissions").first()).toBeNull();
    expect(settled).toBe(false);
    releaseR2!();
    await queueStartedGate;
    expect((await env.RESULTS_DB.prepare("SELECT state FROM submissions").first<{ state: string }>())?.state).toBe("received");
    expect(settled).toBe(false);
    releaseQueue!();
    await expect(pending).resolves.toMatchObject({ state: "queued", duplicate: false });
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

  test("compare-and-transition changes exactly one eligible row", async () => {
    const bindings: IngestEnv = { RESULTS_DB: env.RESULTS_DB, RAW_RESULTS: env.RAW_RESULTS, VALIDATE_QUEUE: { send: async () => undefined } as unknown as Queue };
    const receipt = await receiveEnvelope(bindings, validEnvelope());
    await expect(transition(env.RESULTS_DB, receipt.submission_id, ["queued"], "validating")).resolves.toBe(true);
    await expect(transition(env.RESULTS_DB, receipt.submission_id, ["queued"], "validated")).resolves.toBe(false);
  });
});
