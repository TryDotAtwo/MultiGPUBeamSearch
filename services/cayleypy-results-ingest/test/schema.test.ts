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

  test("rejects claimed solution length mismatch", async () => {
    const envelope = clone(canonicalGolden.cases[0].envelope);
    envelope.solution.length += 1;
    expect(await validateEnvelopeIntegrity(envelope)).toContainEqual({ path: "/solution/length", keyword: "solutionLength" });
  });

  test("rejects a reflected source path whose exact dot path hash is wrong", async () => {
    const envelope = clone(canonicalGolden.cases[1].envelope);
    envelope.orientation.reflected_source_path = ["counterclockwise"];
    expect(await validateEnvelopeIntegrity(envelope)).toContainEqual({ path: "/orientation/reflected_source_sha256", keyword: "reflectionHash" });
  });

  test("replays reflected provenance and rejects a wrong reflected search path", async () => {
    const valid = clone(canonicalGolden.cases[1].envelope);
    expect(await validateEnvelopeIntegrity(valid)).toEqual([]);
    const invalid = clone(canonicalGolden.cases[1].envelope);
    invalid.orientation.searched_path = ["clockwise"];
    expect(await validateEnvelopeIntegrity(invalid)).toContainEqual({ path: "/orientation/searched_path", keyword: "reflectionReplay" });
  });

  test("rejects a reflected provenance inverse which is not the submitted original path", async () => {
    const envelope = clone(canonicalGolden.cases[1].envelope);
    envelope.solution.path = ["counterclockwise"];
    envelope.solution.length = 1;
    expect(await validateEnvelopeIntegrity(envelope)).toContainEqual({ path: "/solution/path", keyword: "reflectionInverse" });
  });

  test("rejects state class overflow, non-integer labels, and a mismatched class count", async () => {
    const envelope = clone(canonicalGolden.cases[0].envelope);
    envelope.proof.initial_state[0] = 3;
    envelope.proof.central_state[1] = 1.5;
    envelope.model.manifest.num_classes = 2;
    const errors = await validateEnvelopeIntegrity(envelope);
    expect(errors).toContainEqual({ path: "/proof/initial_state", keyword: "stateClasses" });
    expect(errors).toContainEqual({ path: "/proof/central_state", keyword: "stateClasses" });
    expect(errors).toContainEqual({ path: "/model/manifest/num_classes", keyword: "stateClasses" });
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