import { validateBatch, validateEnvelopeIntegrity, type ResultEnvelopeV1, type SafeSchemaError } from "./schema.js";
import { validateBatchV2, validateEnvelopeIntegrityV2, type ResultEnvelopeV2 } from "./schema-v2.js";
export type ResultEnvelope = ResultEnvelopeV1 | ResultEnvelopeV2;
export type SchemaVersion = 1 | 2;
export function validateVersionedBatch(value: unknown, version: SchemaVersion, rawByteLength?: number, maxBytes?: number): { ok: true; value: { results: ResultEnvelope[] } } | { ok: false; errors: SafeSchemaError[] } {
  return version === 1 ? validateBatch(value, rawByteLength, maxBytes) : validateBatchV2(value, rawByteLength, maxBytes);
}
export function validateVersionedEnvelope(envelope: ResultEnvelope): Promise<SafeSchemaError[]> {
  return envelope.schema_version === 1 ? validateEnvelopeIntegrity(envelope) : validateEnvelopeIntegrityV2(envelope);
}