import { describe, expect, test } from "vitest";

import canonicalGolden from "../../../configs/cayleypy_results_v1_golden.json";
import { validateBatch, validateEnvelopeIntegrity, type ResultEnvelopeV1 } from "../src/schema.js";

const clone = (value: object): ResultEnvelopeV1 => structuredClone(value) as ResultEnvelopeV1;

describe("canonical CayleyPy results v1 schema", () => {
  test("accepts shared UTF-8, slash, reflected, and empty-source fixtures", async () => {
    expect(canonicalGolden.cases).toHaveLength(3);
    for (const fixture of canonicalGolden.cases) {
      const batch = validateBatch({ schema_version: 1, results: [fixture.envelope] });
      expect(batch).toMatchObject({ ok: true });
      expect(await validateEnvelopeIntegrity(clone(fixture.envelope))).toEqual([]);
    }
  });

  test("rejects a semantic idempotency mismatch", async () => {
    const envelope = clone(canonicalGolden.cases[0].envelope);
    envelope.idempotency_key = "0".repeat(64);
    expect(await validateEnvelopeIntegrity(envelope)).toContainEqual({ path: "/idempotency_key", keyword: "idempotency" });
  });

  test("rejects proof hash, state-length, and permutation mismatches", async () => {
    const envelope = clone(canonicalGolden.cases[0].envelope);
    envelope.proof.initial_state_sha256 = "0".repeat(64);
    envelope.model.manifest.state_len = 2;
    envelope.proof.generators.clockwise = [0, 0, 1];
    const errors = await validateEnvelopeIntegrity(envelope);
    expect(errors).toContainEqual({ path: "/proof/initial_state_sha256", keyword: "proofHash" });
    expect(errors).toContainEqual({ path: "/proof/initial_state", keyword: "stateLength" });
    expect(errors).toContainEqual({ path: "/proof/generators/clockwise", keyword: "permutation" });
  });

  test("rejects a reflected source path whose exact dot path hash is wrong", async () => {
    const envelope = clone(canonicalGolden.cases[1].envelope);
    envelope.orientation.reflected_source_path = ["counterclockwise"];
    expect(await validateEnvelopeIntegrity(envelope)).toContainEqual({ path: "/orientation/reflected_source_sha256", keyword: "reflectionHash" });
  });

  test("rejects output head dimensions inconsistent with the selected profile", async () => {
    const envelope = clone(canonicalGolden.cases[0].envelope);
    envelope.profile.model_class = "output1";
    expect(await validateEnvelopeIntegrity(envelope)).toContainEqual({ path: "/model/manifest/output_dim", keyword: "modelHead" });
  });

  test("rejects batches above the hard four MiB ingress bound", () => {
    const result = validateBatch({ schema_version: 1, results: [canonicalGolden.cases[0].envelope] }, 4 * 1024 * 1024 + 1);
    expect(result).toEqual({ ok: false, errors: [{ path: "", keyword: "maxBytes" }] });
  });
});