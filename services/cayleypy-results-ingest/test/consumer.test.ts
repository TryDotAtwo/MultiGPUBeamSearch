import { env } from "cloudflare:workers";
import canonicalGolden from "../../../configs/cayleypy_results_v1_golden.json";
import { beforeEach, describe, expect, test } from "vitest";

import {
  consumeValidationMessage,
  MAX_QUEUE_ATTEMPTS,
  retryDelaySeconds,
  type ConsumerEnv,
  type QueueMessageLike,
} from "../src/consumer.js";
import { canonicalJson, computeIdempotency, sha256Hex } from "../src/ids.js";
import { replayPath } from "../src/replay.js";
import { validateEnvelopeIntegrity, type ResultEnvelopeV1 } from "../src/schema.js";
import { receiveEnvelope, type IngestEnv } from "../src/storage.js";
import { queue as queueBatch, scheduled, type WorkerEnv } from "../src/worker.js";

function envelope(): ResultEnvelopeV1 {
  return structuredClone(canonicalGolden.cases[0].envelope) as unknown as ResultEnvelopeV1;
}

type ProbeMessage = QueueMessageLike & {
  acked: number;
  retried: number;
  retryOptions: Array<{ delaySeconds?: number } | undefined>;
};

function message(submissionId: string, attempts = 1): ProbeMessage {
  return {
    body: { submission_id: submissionId },
    attempts,
    acked: 0,
    retried: 0,
    retryOptions: [],
    ack() { this.acked += 1; },
    retry(options) { this.retried += 1; this.retryOptions.push(options); },
  };
}

function messageBatch(probe: ProbeMessage): MessageBatch<unknown> {
  return { messages: [probe] } as unknown as MessageBatch<unknown>;
}

function bindings(
  raw: R2Bucket = env.RAW_RESULTS,
  enqueue: (id: string) => Promise<void> = async () => undefined,
): ConsumerEnv {
  return {
    RESULTS_DB: env.RESULTS_DB,
    RAW_RESULTS: raw,
    VALIDATE_QUEUE: { send: async () => undefined } as unknown as Queue,
    VALIDATE_DLQ: { send: async () => undefined } as unknown as Queue,
    GITHUB_WRITER: { getByName: () => ({ enqueueValidated: enqueue }) },
  };
}

async function received(
  inputEnv: ConsumerEnv = bindings(),
  variant = 0,
): Promise<{ id: string; env: ConsumerEnv }> {
  const input = envelope();
  input.puzzle_id += variant;
  input.idempotency_key = await computeIdempotency(input);
  const receipt = await receiveEnvelope(inputEnv as IngestEnv, input);
  return { id: receipt.submission_id, env: inputEnv };
}

async function state(
  id: string,
): Promise<{ state: string; safe_error: string | null; retry_count: number }> {
  return (await env.RESULTS_DB
    .prepare("SELECT state, safe_error, retry_count FROM submissions WHERE submission_id = ?")
    .bind(id)
    .first<{ state: string; safe_error: string | null; retry_count: number }>())!;
}

async function rawKey(id: string): Promise<string> {
  return (await env.RESULTS_DB
    .prepare("SELECT raw_r2_key FROM submissions WHERE submission_id = ?")
    .bind(id)
    .first<string>("raw_r2_key"))!;
}

beforeEach(async () => {
  await env.RESULTS_DB.exec("DELETE FROM submissions");
  await env.RESULTS_DB.exec("DELETE FROM ingest_rate_limits");
  const listed = await env.RAW_RESULTS.list({ prefix: "raw/" });
  if (listed.objects.length) {
    await env.RAW_RESULTS.delete(listed.objects.map((object) => object.key));
  }
});

describe("Task 4 Queue replay consumer", () => {
  test("matches Cloudflare max_retries=8 with nine total delivery attempts", () => {
    expect(MAX_QUEUE_ATTEMPTS).toBe(9);
  });

  test("replays an immutable raw envelope and ACKs after writer enqueue", async () => {
    const seeded = await received();
    const queued = message(seeded.id);
    await consumeValidationMessage(queued, seeded.env, "normal");
    expect(queued).toMatchObject({ acked: 1, retried: 0 });
    expect(await state(seeded.id)).toEqual({
      state: "validated",
      safe_error: null,
      retry_count: 0,
    });
  });

  test("rejects immutable malformed raw without a retry", async () => {
    const seeded = await received();
    const key = await rawKey(seeded.id);
    const raw = "{";
    await env.RAW_RESULTS.put(key, raw, {
      customMetadata: { sha256: await sha256Hex(raw) },
    });
    const queued = message(seeded.id);
    await consumeValidationMessage(queued, seeded.env, "normal");
    expect(queued).toMatchObject({ acked: 1, retried: 0 });
    expect(await state(seeded.id)).toMatchObject({
      state: "rejected",
      safe_error: "invalid_envelope",
    });
  });

  test("missing GitHub writer retains validated state and retries before exhaustion", async () => {
    const seeded = await received();
    const queued = message(seeded.id);
    await consumeValidationMessage(
      queued,
      { ...seeded.env, GITHUB_WRITER: undefined },
      "normal",
    );
    expect(queued).toMatchObject({ acked: 0, retried: 1 });
    expect(await state(seeded.id)).toMatchObject({
      state: "validated",
      safe_error: "publisher_unavailable",
      retry_count: 1,
    });
  });

  test("rejects semantically equivalent raw bytes when the R2 digest mismatches", async () => {
    const seeded = await received();
    const key = await rawKey(seeded.id);
    const original = await env.RAW_RESULTS.get(key);
    const altered = envelope();
    altered.submitted_at = "2026-07-30T00:00:00.000Z";
    await env.RAW_RESULTS.put(key, canonicalJson(altered), {
      customMetadata: { sha256: original!.customMetadata!.sha256! },
    });
    const queued = message(seeded.id);
    await consumeValidationMessage(queued, seeded.env, "normal");
    expect(queued).toMatchObject({ acked: 1, retried: 0 });
    expect(await state(seeded.id)).toMatchObject({
      state: "rejected",
      safe_error: "invalid_envelope",
    });
  });

  test("rejects an oversized raw object before reading its body", async () => {
    const seeded = await received();
    let textCalls = 0;
    const raw = {
      get: async () => ({
        size: 256 * 1024 + 1,
        text: async () => {
          textCalls += 1;
          return "{}";
        },
      }),
    } as unknown as R2Bucket;
    const queued = message(seeded.id);
    await consumeValidationMessage(queued, bindings(raw), "normal");
    expect(queued).toMatchObject({ acked: 1, retried: 0 });
    expect(textCalls).toBe(0);
    expect(await state(seeded.id)).toMatchObject({
      state: "rejected",
      safe_error: "invalid_envelope",
    });
  });

  test("all explicit non-normal modes park before R2 or writer access", async () => {
    for (const [index, mode] of (["store_only", "reject"] as const).entries()) {
      const seeded = await received(bindings(), 10 + index);
      const forbidden = bindings(
        { get: async () => { throw new Error("R2 must not be read while paused"); } } as unknown as R2Bucket,
        async () => { throw new Error("writer must not be called while paused"); },
      );
      const queued = message(seeded.id);
      await consumeValidationMessage(queued, forbidden, mode);
      expect(queued).toMatchObject({ acked: 1, retried: 0 });
      expect(await state(seeded.id)).toEqual({
        state: "retryable",
        safe_error: "ingest_paused",
        retry_count: 0,
      });
    }
  });

  test("worker queue fails closed for missing and unknown modes without R2 access", async () => {
    for (const [index, mode] of ([undefined, "mystery"] as const).entries()) {
      const seeded = await received(bindings(), 20 + index);
      let rawReads = 0;
      const raw = {
        get: async () => {
          rawReads += 1;
          throw new Error("R2 must not be read while paused");
        },
      } as unknown as R2Bucket;
      const workerEnv = {
        ...bindings(raw),
        INGEST_MODE: mode,
        GITHUB_WRITER: {
          getByName: () => { throw new Error("writer must not be called while paused"); },
        },
      } as WorkerEnv;
      const queued = message(seeded.id);
      await queueBatch(messageBatch(queued), workerEnv, {} as ExecutionContext);
      expect(queued).toMatchObject({ acked: 1, retried: 0 });
      expect(rawReads).toBe(0);
      expect(await state(seeded.id)).toMatchObject({
        state: "retryable",
        safe_error: "ingest_paused",
      });
    }
  });

  test("rejects absent and malformed immutable R2 digest metadata", async () => {
    for (const [index, metadata] of (
      [undefined, { sha256: "not-a-sha256" }] as const
    ).entries()) {
      const seeded = await received(bindings(), 30 + index);
      const key = await rawKey(seeded.id);
      const original = await env.RAW_RESULTS.get(key);
      const body = await original!.text();
      if (metadata === undefined) {
        await env.RAW_RESULTS.put(key, body);
      } else {
        await env.RAW_RESULTS.put(key, body, { customMetadata: metadata });
      }
      const queued = message(seeded.id);
      await consumeValidationMessage(queued, seeded.env, "normal");
      expect(queued).toMatchObject({ acked: 1, retried: 0 });
      expect(await state(seeded.id)).toMatchObject({
        state: "rejected",
        safe_error: "invalid_envelope",
      });
    }
  });

  test("malformed and already-terminal deliveries ACK without downstream access", async () => {
    const malformed: ProbeMessage = {
      body: { submission_id: "not-a-uuid" },
      attempts: 1,
      acked: 0,
      retried: 0,
      retryOptions: [],
      ack() { this.acked += 1; },
      retry(options) { this.retried += 1; this.retryOptions.push(options); },
    };
    await consumeValidationMessage(malformed, bindings(), "normal");
    expect(malformed).toMatchObject({ acked: 1, retried: 0 });

    const seeded = await received();
    await env.RESULTS_DB
      .prepare("UPDATE submissions SET state = 'rejected' WHERE submission_id = ?")
      .bind(seeded.id)
      .run();
    const terminal = message(seeded.id);
    await consumeValidationMessage(
      terminal,
      bindings(
        { get: async () => { throw new Error("forbidden"); } } as unknown as R2Bucket,
      ),
      "normal",
    );
    expect(terminal).toMatchObject({ acked: 1, retried: 0 });
  });

  test("enforces logical state bounds and unknown moves deterministically", async () => {
    const oversized = envelope();
    oversized.model.manifest.state_len = 121;
    expect(await validateEnvelopeIntegrity(oversized)).toContainEqual({
      path: "/model/manifest/state_len",
      keyword: "stateLength",
    });
    expect(replayPath([0, 1], ["missing"], { a: [0, 1] }, 2)).toEqual({
      ok: false,
      code: "unknown_move",
    });
    expect(replayPath([0], [], {}, 121)).toEqual({
      ok: false,
      code: "state_length",
    });
  });

  test("transient R2 failures retry with bounded backoff and later validate", async () => {
    const seeded = await received();
    let reads = 0;
    const flakyRaw = {
      get: async (key: string) => {
        reads += 1;
        if (reads === 1) throw new Error("temporary R2 failure");
        return env.RAW_RESULTS.get(key);
      },
    } as unknown as R2Bucket;
    const first = message(seeded.id);
    await consumeValidationMessage(first, bindings(flakyRaw), "normal");
    expect(first).toMatchObject({ acked: 0, retried: 1 });
    expect(first.retryOptions).toEqual([{ delaySeconds: retryDelaySeconds(1) }]);
    expect(await state(seeded.id)).toMatchObject({
      state: "retryable",
      safe_error: "validation_unavailable",
      retry_count: 1,
    });

    const second = message(seeded.id, 2);
    await consumeValidationMessage(second, bindings(), "normal");
    expect(second).toMatchObject({ acked: 1, retried: 0 });
    expect(await state(seeded.id)).toMatchObject({
      state: "validated",
      safe_error: null,
    });
  });

  test("transient D1 lookup retries without acknowledging or reading R2", async () => {
    const seeded = await received();
    const transientDb = {
      prepare: () => { throw new Error("temporary D1 failure"); },
    } as unknown as D1Database;
    const queued = message(seeded.id);
    await consumeValidationMessage(
      queued,
      { ...bindings(), RESULTS_DB: transientDb },
      "normal",
    );
    expect(queued).toMatchObject({ acked: 0, retried: 1 });
    expect(await state(seeded.id)).toMatchObject({
      state: "queued",
      retry_count: 0,
    });
  });

  test("exhaustion records dead_letter and delegates to the configured DLQ", async () => {
    const seeded = await received();
    let dlqSends = 0;
    const unavailable = {
      ...bindings(
        { get: async () => { throw new Error("R2 unavailable"); } } as unknown as R2Bucket,
      ),
      VALIDATE_DLQ: {
        send: async () => { dlqSends += 1; },
      } as unknown as Queue,
    };
    const queued = message(seeded.id, MAX_QUEUE_ATTEMPTS);
    await consumeValidationMessage(queued, unavailable, "normal");
    expect(queued).toMatchObject({ acked: 1, retried: 0 });
    expect(dlqSends).toBe(1);
    expect(await state(seeded.id)).toMatchObject({
      state: "dead_letter",
      safe_error: "validation_unavailable",
      retry_count: 1,
    });
  });

  test("DLQ send failure retries after durable dead_letter and retains raw", async () => {
    const seeded = await received();
    const key = await rawKey(seeded.id);
    const failing = {
      ...bindings(
        { get: async () => { throw new Error("R2 unavailable"); } } as unknown as R2Bucket,
      ),
      VALIDATE_DLQ: {
        send: async () => { throw new Error("DLQ unavailable"); },
      } as unknown as Queue,
    };
    const queued = message(seeded.id, MAX_QUEUE_ATTEMPTS);
    await consumeValidationMessage(queued, failing, "normal");
    expect(queued).toMatchObject({ acked: 0, retried: 1 });
    expect(await state(seeded.id)).toMatchObject({
      state: "dead_letter",
      safe_error: "validation_unavailable",
    });
    expect(await env.RAW_RESULTS.get(key)).not.toBeNull();
  });

  test("duplicate validated delivery repeats idempotent writer enqueue", async () => {
    let calls = 0;
    const flaky = bindings(env.RAW_RESULTS, async () => {
      calls += 1;
      if (calls === 1) throw new Error("writer temporary");
    });
    const seeded = await received(flaky);
    const first = message(seeded.id);
    await consumeValidationMessage(first, flaky, "normal");
    expect(await state(seeded.id)).toMatchObject({
      state: "validated",
      safe_error: "publisher_unavailable",
      retry_count: 1,
    });

    const second = message(seeded.id, 2);
    await consumeValidationMessage(second, flaky, "normal");
    expect(second).toMatchObject({ acked: 1, retried: 0 });
    expect(await state(seeded.id)).toEqual({
      state: "validated",
      safe_error: null,
      retry_count: 1,
    });

    const duplicate = message(seeded.id, 3);
    await consumeValidationMessage(duplicate, flaky, "normal");
    expect(duplicate).toMatchObject({ acked: 1, retried: 0 });
    expect(calls).toBe(3);
  });

  test("writer exhaustion moves the validated row to final DLQ handoff", async () => {
    const unavailableWriter = bindings(
      env.RAW_RESULTS,
      async () => { throw new Error("writer unavailable"); },
    );
    const seeded = await received(unavailableWriter);
    const queued = message(seeded.id, MAX_QUEUE_ATTEMPTS);
    await consumeValidationMessage(queued, unavailableWriter, "normal");
    expect(queued).toMatchObject({ acked: 1, retried: 0 });
    expect(await state(seeded.id)).toMatchObject({
      state: "dead_letter",
      safe_error: "publisher_unavailable",
      retry_count: 1,
    });
  });

  test("paused modes park received, queued, and retryable source states", async () => {
    for (const [index, sourceState] of (
      ["received", "queued", "retryable"] as const
    ).entries()) {
      const seeded = await received(bindings(), 40 + index);
      await env.RESULTS_DB
        .prepare("UPDATE submissions SET state = ? WHERE submission_id = ?")
        .bind(sourceState, seeded.id)
        .run();
      const forbidden = {
        ...bindings(
          { get: async () => { throw new Error("R2 forbidden"); } } as unknown as R2Bucket,
        ),
        GITHUB_WRITER: {
          getByName: () => { throw new Error("writer forbidden"); },
        },
      };
      const queued = message(seeded.id);
      await consumeValidationMessage(queued, forbidden, "store_only");
      expect(queued).toMatchObject({ acked: 1, retried: 0 });
      expect(await state(seeded.id)).toMatchObject({
        state: "retryable",
        safe_error: "ingest_paused",
      });
    }
  });

  test("duplicate validating claimant ACKs without R2 or writer access", async () => {
    const seeded = await received();
    await env.RESULTS_DB
      .prepare("UPDATE submissions SET state = 'validating' WHERE submission_id = ?")
      .bind(seeded.id)
      .run();
    const forbidden = {
      ...bindings(
        { get: async () => { throw new Error("R2 forbidden"); } } as unknown as R2Bucket,
      ),
      GITHUB_WRITER: {
        getByName: () => { throw new Error("writer forbidden"); },
      },
    };
    const queued = message(seeded.id);
    await consumeValidationMessage(queued, forbidden, "normal");
    expect(queued).toMatchObject({ acked: 1, retried: 0 });
    expect(await state(seeded.id)).toMatchObject({ state: "validating" });
  });

  test("D1 park failure retries and normal scheduled recovery resends the same id", async () => {
    const seeded = await received();
    const paused = message(seeded.id);
    await consumeValidationMessage(
      paused,
      {
        ...bindings(),
        RESULTS_DB: {
          prepare: () => { throw new Error("D1 unavailable"); },
        } as unknown as D1Database,
      },
      "store_only",
    );
    expect(paused).toMatchObject({ acked: 0, retried: 1 });

    await env.RESULTS_DB
      .prepare("UPDATE submissions SET state = 'queued', updated_at = ? WHERE submission_id = ?")
      .bind("2000-01-01T00:00:00.000Z", seeded.id)
      .run();
    const sent: unknown[] = [];
    await scheduled(
      { scheduledTime: Date.now() } as ScheduledController,
      {
        ...bindings(),
        INGEST_MODE: "normal",
        VALIDATE_QUEUE: {
          send: async (body: unknown) => { sent.push(body); },
        } as unknown as Queue,
      } as WorkerEnv,
      {} as ExecutionContext,
    );
    expect(sent).toEqual([{ submission_id: seeded.id }]);
    expect((await state(seeded.id)).state).toBe("queued");
  });

  test("scheduled recovery releases stale validating and preserves final states", async () => {
    const validating = await received(bindings(), 51);
    const validated = await received(bindings(), 52);
    const deadLetter = await received(bindings(), 53);
    await env.RESULTS_DB
      .prepare("UPDATE submissions SET state = 'validating', updated_at = ? WHERE submission_id = ?")
      .bind("2000-01-01T00:00:00.000Z", validating.id)
      .run();
    await env.RESULTS_DB
      .prepare("UPDATE submissions SET state = 'validated', updated_at = ? WHERE submission_id = ?")
      .bind("2000-01-01T00:00:00.000Z", validated.id)
      .run();
    await env.RESULTS_DB
      .prepare("UPDATE submissions SET state = 'dead_letter', updated_at = ? WHERE submission_id = ?")
      .bind("2000-01-01T00:00:00.000Z", deadLetter.id)
      .run();
    let sends = 0;
    const workerEnv: WorkerEnv = {
      ...bindings(),
      VALIDATE_QUEUE: {
        send: async () => { sends += 1; },
      } as unknown as Queue,
      INGEST_MODE: "normal",
    };
    await scheduled(
      { scheduledTime: Date.now() } as ScheduledController,
      workerEnv,
      {} as ExecutionContext,
    );
    expect(sends).toBe(1);
    expect((await state(validating.id)).state).toBe("queued");
    expect((await state(validated.id)).state).toBe("validated");
    expect((await state(deadLetter.id)).state).toBe("dead_letter");
  });
});
