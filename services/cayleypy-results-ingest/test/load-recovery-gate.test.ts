import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, test } from "vitest";

import {
  auditRecoverySnapshot,
  parseD1Snapshot,
  parseGitHubSnapshot,
  parseR2Snapshot,
  parseReceiptManifest,
  pollReceiptStatuses,
  type D1AuditRow,
  type GitHubAuditEntry,
  type ReceiptManifestEntry,
  type SubmissionStatus,
} from "./recovery-audit.js";

const loadScriptPath = fileURLToPath(new URL("../load/k6-100-publishers.js", import.meta.url));

const receipt = (
  workloadIndex: number,
  caseKind: "valid" | "duplicate",
  submissionId: string,
  idempotencyKey: string,
): ReceiptManifestEntry => ({
  type: "receipt",
  workload_index: workloadIndex,
  case_kind: caseKind,
  submission_id: submissionId,
  idempotency_key: idempotencyKey,
  status_url: `https://ingest.example/v1/submissions/${submissionId}`,
});

const row = (
  submissionId: string,
  idempotencyKey: string,
  rawKey: string,
  githubPath: string | null,
  state: D1AuditRow["state"] = "published",
): D1AuditRow => ({
  submission_id: submissionId,
  idempotency_key: idempotencyKey,
  state,
  raw_r2_key: rawKey,
  safe_error: state === "rejected"
    ? "invalid_envelope"
    : state === "retryable" || state === "dead_letter"
      ? "github_unavailable"
      : null,
  github_path: githubPath,
});

const status = (
  submissionId: string,
  idempotencyKey: string,
  state: SubmissionStatus["state"] = "published",
): SubmissionStatus => ({
  submission_id: submissionId,
  idempotency_key: idempotencyKey,
  state,
  safe_error: state === "rejected"
    ? "invalid_envelope"
    : state === "retryable" || state === "dead_letter"
      ? "github_unavailable"
      : null,
  retry_count: 0,
  updated_at: "2026-07-29T12:00:00.000Z",
});

describe("Task 8 k6 contract", () => {
  test("pins the exact 100-iteration thresholds and deterministic 80/10/10 mix", () => {
    const source = readFileSync(loadScriptPath, "utf8");
    expect(source).toContain("vus: 100");
    expect(source).toContain("iterations: 100");
    expect(source).toContain('http_req_failed: ["rate<0.01"]');
    expect(source).toContain('http_req_duration: ["p(95)<2000"]');
    expect(source).toContain('checks: ["rate>0.99"]');
    expect(source).toContain("const UNIQUE_VALID_RESULTS = 80");
    expect(source).toContain("const DUPLICATE_RESULTS = 10");
    expect(source).toContain("const INVALID_RESULTS = 10");
    expect(source).toContain("exec.scenario.iterationInTest");
  });

  test("uses only the environment endpoint, bounded 429 retries, and safe receipt logs", () => {
    const source = readFileSync(loadScriptPath, "utf8");
    expect(source).toContain("__ENV.INGEST_BASE_URL");
    expect(source).toContain("MAX_429_RETRIES");
    expect(source).toContain("Retry-After");
    expect(source).toContain("http.expectedStatuses");
    expect(source).toContain("CAYLEYPY_RECEIPT");
    expect(source).not.toMatch(/Authorization|GITHUB_TOKEN|KAGGLE_KEY|PRIVATE_KEY/);
    expect(source).not.toMatch(/console\.(?:log|error)\([^\n]*(?:envelope|requestBody|response\.body)/);
  });
});

describe("bounded recovery manifest parsing", () => {
  test("accepts plain and k6 JSON safe receipt lines without retaining envelopes", () => {
    const first = receipt(0, "valid", "018f7a24-8f6b-7c8e-9d1b-000000000001", "a".repeat(64));
    const second = receipt(80, "duplicate", first.submission_id, first.idempotency_key);
    const input = [
      `CAYLEYPY_RECEIPT\t${JSON.stringify(first)}`,
      JSON.stringify({ level: "info", msg: `CAYLEYPY_RECEIPT\t${JSON.stringify(second)}` }),
      "unrelated k6 output",
    ].join("\n");

    expect(parseReceiptManifest(input)).toEqual([first, second]);
  });

  test("rejects oversized or unbounded manifests before parsing payload data", () => {
    const entry = receipt(0, "valid", "018f7a24-8f6b-7c8e-9d1b-000000000001", "a".repeat(64));
    expect(() => parseReceiptManifest("x".repeat(1_048_577))).toThrow("manifest_too_large");
    expect(() => parseReceiptManifest(
      Array.from({ length: 201 }, () => `CAYLEYPY_RECEIPT\t${JSON.stringify(entry)}`).join("\n"),
    )).toThrow("manifest_receipt_limit");
  });
});

describe("bounded status polling", () => {
  test("fetches every unique receipt from the configured endpoint without following manifest hosts", async () => {
    const first = receipt(0, "valid", "018f7a24-8f6b-7c8e-9d1b-000000000001", "a".repeat(64));
    const duplicate = { ...receipt(80, "duplicate", first.submission_id, first.idempotency_key), status_url: `https://untrusted.example/v1/submissions/${first.submission_id}` };
    const second = receipt(1, "valid", "018f7a24-8f6b-7c8e-9d1b-000000000002", "b".repeat(64));
    const requested: string[] = [];
    const fetchImpl = async (input: RequestInfo | URL): Promise<Response> => {
      const url = new URL(String(input));
      requested.push(url.toString());
      const submissionId = url.pathname.split("/").at(-1)!;
      const idempotencyKey = submissionId === first.submission_id
        ? first.idempotency_key
        : second.idempotency_key;
      return new Response(JSON.stringify(status(submissionId, idempotencyKey)), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    };

    await expect(pollReceiptStatuses(
      [first, duplicate, second],
      "https://configured.example/base",
      { requirePublished: true, timeoutMs: 0, fetchImpl: fetchImpl as typeof fetch },
    )).resolves.toEqual([
      status(first.submission_id, first.idempotency_key),
      status(second.submission_id, second.idempotency_key),
    ]);
    expect(requested).toEqual([
      `https://configured.example/v1/submissions/${first.submission_id}`,
      `https://configured.example/v1/submissions/${second.submission_id}`,
    ]);
  });

  test("fails with a bounded timeout when a receipt status remains unavailable", async () => {
    const first = receipt(0, "valid", "018f7a24-8f6b-7c8e-9d1b-000000000001", "a".repeat(64));
    await expect(pollReceiptStatuses([first], "https://configured.example", {
      requirePublished: true,
      timeoutMs: 0,
      fetchImpl: (async () => new Response(null, { status: 503 })) as typeof fetch,
    })).rejects.toThrow("status_poll_timeout");
  });
});

describe("recovery snapshot normalization", () => {
  test("normalizes bounded Wrangler D1, R2, and GitHub evidence shapes", () => {
    const d1 = row(
      "018f7a24-8f6b-7c8e-9d1b-000000000001",
      "a".repeat(64),
      "raw/v1/2026/07/29/one.json",
      "results/v1/toy/cycle-3/2026/07/29/one.json",
    );
    expect(parseD1Snapshot(JSON.stringify([{ results: [d1] }]))).toEqual([d1]);
    expect(parseR2Snapshot(JSON.stringify({ objects: [{ key: d1.raw_r2_key }] }))).toEqual([d1.raw_r2_key]);
    const github: GitHubAuditEntry = {
      path: d1.github_path!,
      submission_id: d1.submission_id,
      idempotency_key: d1.idempotency_key,
    };
    expect(parseGitHubSnapshot(JSON.stringify({ entries: [github] }))).toEqual([github]);
  });
});

describe("recovery invariants", () => {
  const submissionOne = "018f7a24-8f6b-7c8e-9d1b-000000000001";
  const submissionTwo = "018f7a24-8f6b-7c8e-9d1b-000000000002";
  const keyOne = "a".repeat(64);
  const keyTwo = "b".repeat(64);
  const rawOne = "raw/v1/2026/07/29/one.json";
  const rawTwo = "raw/v1/2026/07/29/two.json";
  const pathOne = "results/v1/toy/cycle-3/2026/07/29/one.json";
  const pathTwo = "results/v1/toy/cycle-3/2026/07/29/two.json";

  const validInput = () => ({
    receipts: [
      receipt(0, "valid", submissionOne, keyOne),
      receipt(80, "duplicate", submissionOne, keyOne),
      receipt(1, "valid", submissionTwo, keyTwo),
    ],
    d1Rows: [
      row(submissionOne, keyOne, rawOne, pathOne),
      row(submissionTwo, keyTwo, rawTwo, pathTwo),
    ],
    r2Keys: [rawOne, rawTwo],
    githubEntries: [
      { path: pathOne, submission_id: submissionOne, idempotency_key: keyOne },
      { path: pathTwo, submission_id: submissionTwo, idempotency_key: keyTwo },
    ],
    statuses: [
      status(submissionOne, keyOne),
      status(submissionTwo, keyTwo),
    ],
    requirePublished: true,
  });

  test("proves duplicate convergence, raw durability, status agreement, and one GitHub file per key", () => {
    expect(auditRecoverySnapshot(validInput())).toEqual({
      ok: true,
      errors: [],
      summary: {
        receipt_events: 3,
        unique_receipts: 2,
        duplicate_events: 1,
        published: 2,
        rejected: 0,
        recoverable: 0,
        github_files: 2,
      },
    });
  });

  test("fails closed when an accepted receipt has no immutable raw object", () => {
    const input = validInput();
    input.r2Keys = [rawOne];
    const result = auditRecoverySnapshot(input);
    expect(result.ok).toBe(false);
    expect(result.errors).toContain(`accepted_missing_raw:${submissionTwo}`);
  });

  test("rejects two repository files for one semantic idempotency key", () => {
    const input = validInput();
    input.githubEntries.push({
      path: `${pathOne}.duplicate`,
      submission_id: submissionOne,
      idempotency_key: keyOne,
    });
    const result = auditRecoverySnapshot(input);
    expect(result.ok).toBe(false);
    expect(result.errors).toContain(`github_duplicate_idempotency:${keyOne}`);
  });

  test("requires validated load receipts to publish after outage recovery", () => {
    const input = validInput();
    input.d1Rows[1] = row(submissionTwo, keyTwo, rawTwo, null, "retryable");
    input.statuses[1] = status(submissionTwo, keyTwo, "retryable");
    input.githubEntries.pop();
    const strict = auditRecoverySnapshot(input);
    expect(strict.ok).toBe(false);
    expect(strict.errors).toContain(`validated_not_published:${submissionTwo}:retryable`);

    input.requirePublished = false;
    expect(auditRecoverySnapshot(input)).toMatchObject({
      ok: true,
      summary: { published: 1, recoverable: 1 },
    });
  });

  test("requires terminal rejected receipts to retain a safe reason", () => {
    const input = validInput();
    input.d1Rows[1] = { ...row(submissionTwo, keyTwo, rawTwo, null, "rejected"), safe_error: null };
    input.statuses[1] = { ...status(submissionTwo, keyTwo, "rejected"), safe_error: null };
    input.githubEntries.pop();
    input.requirePublished = false;
    const result = auditRecoverySnapshot(input);
    expect(result.ok).toBe(false);
    expect(result.errors).toContain(`rejected_missing_reason:${submissionTwo}`);
  });
});
