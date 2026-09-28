import { expect, test } from "vitest";
import fixture from "../../../configs/cube555_publication_test.json";
import { validateBatch, validateEnvelopeIntegrity, type ResultEnvelopeV1 } from "../src/schema.js";
test("accepts Python-produced 150-facelet proof with high-index move", async () => {
 expect(validateBatch({schema_version:1,results:[fixture]}).ok).toBe(true);
 expect(await validateEnvelopeIntegrity(fixture as ResultEnvelopeV1)).toEqual([]);
});
test("rejects tampered high-index proof", async () => {
 const e=structuredClone(fixture) as ResultEnvelopeV1;
 e.proof.initial_state[149]=149;
 expect((await validateEnvelopeIntegrity(e)).length).toBeGreaterThan(0);
});
