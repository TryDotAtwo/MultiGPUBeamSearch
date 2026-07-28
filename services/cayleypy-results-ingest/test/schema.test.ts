import { describe, expect, test } from "vitest";

import { validateBatch } from "../src/schema.js";

const validResult = () => ({
  schema_version: 1,
  submission_id: "018f7a24-8f6b-7c8e-9d1b-2a3b4c5d6e7f",
  run_id: "run-20260728-001",
  idempotency_key: "a".repeat(64),
  author: { name: "Ada Lovelace", verification: "claimed" },
  kaggle: { owner: "ada", slug: "cayleypy-run", version: 1 },
  competition: "santa-2023",
  puzzle_type: "cube",
  puzzle_id: 42,
  proof: {
    initial_state: [0, 1, 2],
    central_state: [1, 2, 0],
    generators: { r: [1, 2, 0] },
  },
  orientation: { search_mode: "off", final_orientation: "original" },
  solution: { path: ["r"], length: 1, solved_depth: 1, validation: "valid" },
  profile: { requested_beam: 1024, effective_beam: 1024, alignment_delta: 0, evidence: "t4-v1" },
  runtime: { touch_bfs_radius: 1, solution_mode: "first", max_depth: 10, max_collected_solutions: 1 },
  model: { filename: "model.pth", sha256: "b".repeat(64), format: "batchnorm-folded", manifest: { output_dim: 1 } },
  hardware: { gpu_names: ["Tesla T4"], world_size: 2, total_runtime_ms: 1234 },
  solver_commit: "c".repeat(40),
  submitted_at: "2026-07-28T10:00:00.000Z",
});

const validate = (result: unknown) => validateBatch({ schema_version: 1, results: [result] });

describe("validateBatch", () => {
  test("accepts a bounded version-one result envelope", () => {
    const result = validate(validResult());
    expect(result).toEqual({ ok: true, value: { schema_version: 1, results: [validResult()] } });
  });

  test("rejects an unknown envelope field without echoing its value", () => {
    const result = validate({ ...validResult(), unexpected: "private-token-value" });
    expect(result).toMatchObject({ ok: false, errors: [{ path: "", keyword: "additionalProperties" }] });
    expect(JSON.stringify(result)).not.toContain("private-token-value");
  });

  test("rejects an unsupported schema version", () => {
    const result = validateBatch({ schema_version: 2, results: [validResult()] });
    expect(result).toMatchObject({ ok: false, errors: [{ path: "/schema_version", keyword: "const" }] });
  });

  test("rejects oversized author names, paths, and proofs", () => {
    for (const result of [
      { ...validResult(), author: { name: "a".repeat(129), verification: "claimed" } },
      { ...validResult(), solution: { ...validResult().solution, path: Array(4097).fill("r") } },
      { ...validResult(), proof: { ...validResult().proof, initial_state: Array(129).fill(0) } },
    ]) {
      expect(validate(result)).toMatchObject({ ok: false });
    }
  });

  test("rejects invalid closed enums", () => {
    const result = validate({ ...validResult(), runtime: { ...validResult().runtime, solution_mode: "unbounded" } });
    expect(result).toMatchObject({ ok: false, errors: [{ path: "/runtime/solution_mode", keyword: "enum" }] });
  });

  test("rejects batches with more than one hundred results", () => {
    const result = validateBatch({ schema_version: 1, results: Array.from({ length: 101 }, validResult) });
    expect(result).toMatchObject({ ok: false, errors: [{ path: "/results", keyword: "maxItems" }] });
  });

  test("rejects a serialized request larger than twenty-five MiB", () => {
    const result = validateBatch({ schema_version: 1, results: [validResult()], padding: "x".repeat(25 * 1024 * 1024) });
    expect(result).toMatchObject({ ok: false });
  });
});
