import { env } from "cloudflare:workers";
import {
  evictDurableObject,
  runDurableObjectAlarm,
  runInDurableObject,
} from "cloudflare:test";
import { exportPKCS8, generateKeyPair } from "jose";
import {
  afterEach,
  beforeAll,
  describe,
  expect,
  test,
  vi,
} from "vitest";

import canonicalGolden from "../../../configs/cayleypy_results_v1_golden.json";
import { resetInstallationTokenCacheForTest } from "../src/github-app.js";
import {
  GitHubWriter,
  branchRoute,
  resultPath,
  resolveWriterMode,
} from "../src/github-writer.js";
import { canonicalJson, computeIdempotency } from "../src/ids.js";
import type { ResultEnvelopeV1 } from "../src/schema.js";
import { receiveEnvelope, type IngestEnv } from "../src/storage.js";

const ID = "01820000-0000-7000-8000-000000000001";
const RECOVERY_ID = "01820000-0000-7000-8000-000000000002";
const HEAD_SHA = "1".repeat(40);
const BASE_TREE_SHA = "2".repeat(40);
const NEW_TREE_SHA = "3".repeat(40);
const NEW_COMMIT_SHA = "4".repeat(40);

interface Seeded {
  submissionId: string;
  envelope: ResultEnvelopeV1;
  rawKey: string;
}

interface RouteCall {
  method: string;
  url: URL;
  body: unknown;
}

type ExistingMode = "absent" | "same" | "different";

let privateKey = "";

beforeAll(async () => {
  const pair = await generateKeyPair("RS256", {
    modulusLength: 2048,
    extractable: true,
  });
  privateKey = await exportPKCS8(pair.privateKey);
});

afterEach(() => {
  vi.unstubAllGlobals();
  resetInstallationTokenCacheForTest();
});

function pendingKey(value: string): string {
  return `pending/${value}`;
}

function stub(name = "writer-test") {
  return env.GITHUB_WRITER.get(env.GITHUB_WRITER.idFromName(name));
}

function mutableWriterEnv(instance: unknown): Record<string, unknown> {
  return (instance as { env: Record<string, unknown> }).env;
}

function configureWriter(
  instance: unknown,
  overrides: Record<string, unknown> = {},
): void {
  Object.assign(mutableWriterEnv(instance), {
    INGEST_MODE: "normal",
    GITHUB_APP_ID: "12345",
    GITHUB_APP_INSTALLATION_ID: "67890",
    GITHUB_APP_PRIVATE_KEY: privateKey.replace(/\n/g, "\\n"),
    GITHUB_API_URL: "https://github.example",
    REPO_OWNER: "TryDotAtwo",
    REPO_NAME: "cayleypy-beam-results",
    STAGING_BRANCH: "ingest/staging",
    ...overrides,
  });
}

async function seedValidated(variant: number): Promise<Seeded> {
  const envelope = structuredClone(
    canonicalGolden.cases[0].envelope,
  ) as ResultEnvelopeV1;
  envelope.puzzle_id += variant;
  envelope.idempotency_key = await computeIdempotency(envelope);
  const receipt = await receiveEnvelope(
    {
      RESULTS_DB: env.RESULTS_DB,
      RAW_RESULTS: env.RAW_RESULTS,
      VALIDATE_QUEUE: {
        send: async () => undefined,
      } as unknown as Queue,
    } as IngestEnv,
    envelope,
  );
  await env.RESULTS_DB
    .prepare(
      "UPDATE submissions SET state = 'validated' WHERE submission_id = ?",
    )
    .bind(receipt.submission_id)
    .run();
  const row = await env.RESULTS_DB
    .prepare(
      "SELECT raw_r2_key FROM submissions WHERE submission_id = ?",
    )
    .bind(receipt.submission_id)
    .first<{ raw_r2_key: string }>();
  if (row === null) throw new Error("seed_failed");
  return {
    submissionId: receipt.submission_id,
    envelope,
    rawKey: row.raw_r2_key,
  };
}

async function pending(name: string): Promise<string[]> {
  return runInDurableObject(stub(name), async (_instance, state) => [
    ...(await state.storage.list({ prefix: "pending/" })).keys(),
  ]);
}

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function utf8Base64(value: string): string {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function githubRouter(
  existing: ExistingMode,
  expectedBody: string,
): { fetcher: ReturnType<typeof vi.fn>; calls: RouteCall[] } {
  const calls: RouteCall[] = [];
  const fetcher = vi.fn(async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    const url = new URL(String(input));
    const method = init?.method ?? "GET";
    let body: unknown;
    if (init?.body !== undefined) body = JSON.parse(String(init.body));
    calls.push({ method, url, body });

    if (
      method === "POST" &&
      url.pathname === "/app/installations/67890/access_tokens"
    ) {
      return json({
        token: `ghs_ephemeral_${existing}`,
        expires_at: new Date(Date.now() + 5 * 60_000).toISOString(),
      }, 201);
    }
    if (
      method === "GET" &&
      url.pathname ===
        "/repos/TryDotAtwo/cayleypy-beam-results/git/ref/heads/ingest/staging"
    ) {
      return json({
        ref: "refs/heads/ingest/staging",
        object: { type: "commit", sha: HEAD_SHA },
      });
    }
    if (
      method === "GET" &&
      url.pathname ===
        `/repos/TryDotAtwo/cayleypy-beam-results/git/commits/${HEAD_SHA}`
    ) {
      return json({ sha: HEAD_SHA, tree: { sha: BASE_TREE_SHA } });
    }
    if (
      method === "GET" &&
      url.pathname.startsWith(
        "/repos/TryDotAtwo/cayleypy-beam-results/contents/results/v1/",
      )
    ) {
      expect(url.searchParams.get("ref")).toBe("ingest/staging");
      if (existing === "absent") return new Response(null, { status: 404 });
      return json({
        content: utf8Base64(
          existing === "same" ? expectedBody : '{"different":true}',
        ),
      });
    }
    if (
      method === "POST" &&
      url.pathname ===
        "/repos/TryDotAtwo/cayleypy-beam-results/git/trees"
    ) {
      return json({ sha: NEW_TREE_SHA }, 201);
    }
    if (
      method === "POST" &&
      url.pathname ===
        "/repos/TryDotAtwo/cayleypy-beam-results/git/commits"
    ) {
      return json({ sha: NEW_COMMIT_SHA }, 201);
    }
    if (
      method === "PATCH" &&
      url.pathname ===
        "/repos/TryDotAtwo/cayleypy-beam-results/git/refs/heads/ingest/staging"
    ) {
      return json({ object: { sha: NEW_COMMIT_SHA } });
    }
    throw new Error(`unexpected_github_route:${method}:${url.pathname}`);
  });
  return { fetcher, calls };
}

async function flushWithRouter(
  name: string,
  seeded: Seeded,
  existing: ExistingMode,
) {
  const expectedBody = canonicalJson({
    submission_id: seeded.submissionId,
    envelope: seeded.envelope,
  });
  const router = githubRouter(existing, expectedBody);
  vi.stubGlobal("fetch", router.fetcher);
  const target = stub(name);
  await target.enqueueValidated(seeded.submissionId);
  const outcome = await runInDurableObject(
    target,
    async (instance, state) => {
      const writer = instance as unknown as GitHubWriter;
      configureWriter(writer);
      resetInstallationTokenCacheForTest();
      const result = await writer.flush();
      return {
        result,
        pending: [
          ...(await state.storage.list({ prefix: "pending/" })).keys(),
        ],
      };
    },
  );
  return { ...router, ...outcome, expectedBody };
}

describe("GitHubWriter pure contracts", () => {
  test("resolves only the three exact ingest modes", () => {
    expect(resolveWriterMode("normal")).toBe("normal");
    expect(resolveWriterMode("store_only")).toBe("store_only");
    expect(resolveWriterMode("reject")).toBe("reject");
    for (const value of [undefined, "", "NORMAL", "paused", {}, 1]) {
      expect(resolveWriterMode(value)).toBe("reject");
    }
  });

  test("builds a normalized append-only path from the server id", () => {
    const record = {
      competition: "Santa 2026",
      puzzle_type: "Mega Minx",
      puzzle_id: 7,
      submitted_at: "2026-07-29T00:00:00.000Z",
    } as ResultEnvelopeV1;
    expect(resultPath(ID, record)).toBe(
      `results/v1/santa-2026/mega-minx/7/2026-07-29/${ID}.json`,
    );
    expect(() => resultPath("client-path", record)).toThrow(
      "github_path_invalid",
    );
  });

  test("validates and routes a slash-separated staging branch", () => {
    expect(branchRoute("ingest/staging")).toBe("ingest/staging");
    expect(branchRoute("release/v1.2")).toBe("release/v1.2");
    for (const value of [undefined, "", "/main", "main/", "a//b", "a/../b"]) {
      expect(() => branchRoute(value)).toThrow("github_branch_invalid");
    }
  });
});

describe("GitHubWriter real Durable Object harness", () => {
  test("100 concurrent duplicate RPC enqueues collapse to 100 durable records", async () => {
    const target = stub("concurrent");
    const ids = Array.from(
      { length: 100 },
      (_, n) =>
        `01820000-0000-7000-8000-${String(n).padStart(12, "0")}`,
    );
    await Promise.all(
      [...ids, ...ids].map((submissionId) =>
        target.enqueueValidated(submissionId)
      ),
    );
    expect(await pending("concurrent")).toHaveLength(100);
    expect(
      await runInDurableObject(
        target,
        async (_instance, state) => state.storage.getAlarm(),
      ),
    ).not.toBeNull();
  });

  test("paused flush retains work and makes zero external requests", async () => {
    const name = "paused";
    const target = stub(name);
    await target.enqueueValidated(ID);
    const fetcher = vi.fn();
    vi.stubGlobal("fetch", fetcher);
    const outcome = await runInDurableObject(
      target,
      async (instance, state) => {
        const writer = instance as unknown as GitHubWriter;
        mutableWriterEnv(writer).INGEST_MODE = "store_only";
        const result = await writer.flush();
        return {
          result,
          pending: [
            ...(await state.storage.list({ prefix: "pending/" })).keys(),
          ],
          alarm: await state.storage.getAlarm(),
        };
      },
    );
    expect(outcome.result).toEqual({ staged: 0, retained: 1 });
    expect(outcome.pending).toEqual([pendingKey(ID)]);
    expect(outcome.alarm).not.toBeNull();
    expect(fetcher).not.toHaveBeenCalled();
  });

  test("eviction reconstructs the alarm and normal recovery drops a missing D1 id", async () => {
    const name = "eviction";
    const target = stub(name);
    await target.enqueueValidated(RECOVERY_ID);
    await runInDurableObject(
      target,
      async (_instance, state) => state.storage.deleteAlarm(),
    );
    await evictDurableObject(target);
    const fetcher = vi.fn();
    vi.stubGlobal("fetch", fetcher);
    const reconstructed = await runInDurableObject(
      target,
      async (instance, state) => {
        const writer = instance as unknown as GitHubWriter;
        mutableWriterEnv(writer).INGEST_MODE = "normal";
        return state.storage.getAlarm();
      },
    );
    expect(reconstructed).not.toBeNull();
    await runDurableObjectAlarm(target);
    expect(await pending(name)).toEqual([]);
    expect(
      await runInDurableObject(
        target,
        async (_instance, state) => state.storage.getAlarm(),
      ),
    ).toBeNull();
    expect(fetcher).not.toHaveBeenCalled();
  });

  test("tampered real R2 raw is terminalized and retained for forensics", async () => {
    const seeded = await seedValidated(11);
    await env.RAW_RESULTS.put(seeded.rawKey, "{}", {
      customMetadata: { sha256: "0".repeat(64) },
    });
    const name = "tamper";
    const target = stub(name);
    await target.enqueueValidated(seeded.submissionId);
    const fetcher = vi.fn();
    vi.stubGlobal("fetch", fetcher);
    await runInDurableObject(target, async (instance) => {
      const writer = instance as unknown as GitHubWriter;
      mutableWriterEnv(writer).INGEST_MODE = "normal";
      await writer.flush();
    });
    expect(
      await env.RESULTS_DB
        .prepare(
          "SELECT state, safe_error FROM submissions WHERE submission_id = ?",
        )
        .bind(seeded.submissionId)
        .first(),
    ).toMatchObject({
      state: "dead_letter",
      safe_error: "publication_raw_integrity_invalid",
    });
    expect(await env.RAW_RESULTS.get(seeded.rawKey)).not.toBeNull();
    expect(await pending(name)).toEqual([]);
    expect(fetcher).not.toHaveBeenCalled();
  });

  test("publishes a valid record through token, tree, commit, and ref routes", async () => {
    const seeded = await seedValidated(21);
    const outcome = await flushWithRouter("github-add", seeded, "absent");
    expect(outcome.result).toEqual({ staged: 1, retained: 0 });
    expect(outcome.pending).toEqual([]);
    expect(
      await env.RESULTS_DB
        .prepare("SELECT submission_id FROM submissions WHERE submission_id = ?")
        .bind(seeded.submissionId)
        .first(),
    ).toBeNull();
    const tree = outcome.calls.find(
      (call) =>
        call.method === "POST" && call.url.pathname.endsWith("/git/trees"),
    );
    expect(tree?.body).toMatchObject({
      base_tree: BASE_TREE_SHA,
      tree: [{
        path: resultPath(seeded.submissionId, seeded.envelope),
        content: outcome.expectedBody,
      }],
    });
    expect(
      outcome.calls.map(
        (call) => `${call.method} ${call.url.pathname}`,
      ),
    ).toEqual([
      "POST /app/installations/67890/access_tokens",
      "GET /repos/TryDotAtwo/cayleypy-beam-results/git/ref/heads/ingest/staging",
      `GET /repos/TryDotAtwo/cayleypy-beam-results/git/commits/${HEAD_SHA}`,
      expect.stringMatching(
        /^GET \/repos\/TryDotAtwo\/cayleypy-beam-results\/contents\/results\/v1\//,
      ),
      "POST /repos/TryDotAtwo/cayleypy-beam-results/git/trees",
      "POST /repos/TryDotAtwo/cayleypy-beam-results/git/commits",
      "PATCH /repos/TryDotAtwo/cayleypy-beam-results/git/refs/heads/ingest/staging",
    ]);
    expect(await env.RAW_RESULTS.get(seeded.rawKey)).toBeNull();
  });

  test("reconciles identical remote content without a GitHub write", async () => {
    const seeded = await seedValidated(22);
    const outcome = await flushWithRouter("github-same", seeded, "same");
    expect(outcome.result).toEqual({ staged: 1, retained: 0 });
    expect(outcome.pending).toEqual([]);
    expect(
      await env.RESULTS_DB
        .prepare("SELECT submission_id FROM submissions WHERE submission_id = ?")
        .bind(seeded.submissionId)
        .first(),
    ).toBeNull();
    expect(await env.RAW_RESULTS.get(seeded.rawKey)).toBeNull();
    expect(
      outcome.calls.some(
        (call) =>
          call.url.pathname.endsWith("/git/trees") ||
          (call.method === "POST" &&
            call.url.pathname.endsWith("/git/commits")) ||
          call.method === "PATCH",
      ),
    ).toBe(false);
  });

  test("terminalizes different remote content without overwriting it", async () => {
    const seeded = await seedValidated(23);
    const outcome = await flushWithRouter(
      "github-different",
      seeded,
      "different",
    );
    expect(outcome.result).toEqual({ staged: 0, retained: 0 });
    expect(outcome.pending).toEqual([]);
    expect(
      await env.RESULTS_DB
        .prepare(
          "SELECT state, safe_error, github_path, github_commit_sha FROM submissions WHERE submission_id = ?",
        )
        .bind(seeded.submissionId)
        .first(),
    ).toEqual({
      state: "dead_letter",
      safe_error: "publication_path_conflict",
      github_path: null,
      github_commit_sha: null,
    });
    expect(
      outcome.calls.some(
        (call) =>
          call.url.pathname.endsWith("/git/trees") ||
          (call.method === "POST" &&
            call.url.pathname.endsWith("/git/commits")) ||
          call.method === "PATCH",
      ),
    ).toBe(false);
    expect(await env.RAW_RESULTS.get(seeded.rawKey)).not.toBeNull();
  });
});

test(
  "drops 100 stale poison keys so a later validated record is eventually reachable",
  async () => {
    const name = "poison";
    const target = stub(name);
    await Promise.all(Array.from({ length: 100 }, (_, n) => target.enqueueValidated(
      `01820000-0000-7000-8000-${String(n).padStart(12, "0")}`,
    )));
    const seeded = await seedValidated(91);
    const expected = canonicalJson({ submission_id: seeded.submissionId, envelope: seeded.envelope });
    const router = githubRouter("same", expected);
    vi.stubGlobal("fetch", router.fetcher);
    await target.enqueueValidated(seeded.submissionId);
    await runInDurableObject(target, async (instance) => {
      const writer = instance as unknown as GitHubWriter;
      configureWriter(writer);
      await writer.flush();
      await writer.flush();
      await writer.flush();
    });
    expect(await pending(name)).toEqual([]);
    expect(
      await env.RESULTS_DB
        .prepare("SELECT state FROM submissions WHERE submission_id = ?")
        .bind(seeded.submissionId)
        .first(),
    ).toBeNull();
    expect(await env.RAW_RESULTS.get(seeded.rawKey)).toBeNull();
  },
  15_000,
);

test(
  "keeps one GitHub publication flush within the free Worker subrequest budget",
  async () => {
    const name = "free-subrequest-budget";
    const target = stub(name);
    const seeded = await Promise.all(
      Array.from({ length: 41 }, (_, n) => seedValidated(1_000 + n)),
    );
    await Promise.all(
      seeded.map((item) => target.enqueueValidated(item.submissionId)),
    );
    const router = githubRouter("absent", "");
    vi.stubGlobal("fetch", router.fetcher);

    const result = await runInDurableObject(target, async (instance) => {
      const writer = instance as unknown as GitHubWriter;
      configureWriter(writer);
      resetInstallationTokenCacheForTest();
      return writer.flush();
    });

    const preflightCalls = router.calls.filter((call) =>
      call.method === "GET" && call.url.pathname.includes("/contents/results/v1/")
    );
    expect(preflightCalls).toHaveLength(40);
    expect(result).toEqual({ staged: 40, retained: 1 });
    expect(await pending(name)).toHaveLength(1);
  },
  15_000,
);

test("transient alarm failure forcibly leaves a future alarm", async () => {
  const seeded = await seedValidated(92);
  const target = stub("retry-alarm");
  await target.enqueueValidated(seeded.submissionId);
  vi.stubGlobal("fetch", vi.fn(async () => {
    throw new Error("network_down");
  }));
  await expect(
    runInDurableObject(target, async (instance) => {
      const writer = instance as unknown as GitHubWriter;
      configureWriter(writer);
      await writer.alarm();
    }),
  ).rejects.toThrow("github_writer_retryable");
  expect(
    await runInDurableObject(
      target,
      async (_instance, state) => state.storage.getAlarm(),
    ),
  ).not.toBeNull();
  expect(
    await env.RESULTS_DB
      .prepare("SELECT state FROM submissions WHERE submission_id = ?")
      .bind(seeded.submissionId)
      .first(),
  ).toEqual({ state: "validated" });
  expect(await env.RAW_RESULTS.get(seeded.rawKey)).not.toBeNull();
  expect(await pending("retry-alarm")).toEqual([pendingKey(seeded.submissionId)]);
});
