import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import schema from "../../../configs/cayleypy_results_schema_v1.json";

export interface ResultEnvelopeV1 {
  schema_version: 1;
  submission_id: string;
  run_id: string;
  idempotency_key: string;
  author: { name: string; kaggle_username?: string; verification: "claimed" };
  kaggle: { owner: string; slug: string; version: number; run_url?: string };
  competition: string;
  puzzle_type: string;
  puzzle_id: number;
  proof: { initial_state: number[]; central_state: number[]; generators: Record<string, number[]> };
  orientation: { search_mode: "off" | "after_original" | "only"; final_orientation: "original" | "reflected"; reflected_source_id?: string; reflected_path_hash?: string };
  solution: { path: string[]; length: number; solved_depth: number; touch_depth?: number; validation: "valid"; collection_index?: number; collection_status?: "first" | "complete" | "capacity_reached" };
  profile: { requested_beam: number; effective_beam: number; alignment_delta: number; selected_profile?: string; evidence: string };
  runtime: { touch_bfs_radius: number; solution_mode: "first" | "collect"; max_depth: number; max_collected_solutions: number };
  model: { filename: string; sha256: string; format: "batchnorm-folded" | "resmlp-layernorm"; manifest: Record<string, string | number | boolean | null> };
  hardware: { gpu_names: string[]; world_size: number; total_runtime_ms: number };
  solver_commit: string;
  submitted_at: string;
}

export interface ResultBatchV1 { schema_version: 1; results: ResultEnvelopeV1[] }
export interface SafeSchemaError { path: string; keyword: string }
export type ValidationResult = { ok: true; value: ResultBatchV1 } | { ok: false; errors: SafeSchemaError[] };

const MAX_SERIALIZED_BATCH_BYTES = 25 * 1024 * 1024;
const ajv = new Ajv2020({ allErrors: true, strict: true, allowUnionTypes: true });
addFormats(ajv);
const validate = ajv.compile(schema);

function isResultBatchV1(value: unknown): value is ResultBatchV1 {
  return validate(value);
}

export function validateBatch(value: unknown, rawByteLength?: number): ValidationResult {
  let serialized: string;
  try { serialized = JSON.stringify(value); } catch { return { ok: false, errors: [{ path: "", keyword: "serialization" }] }; }
  const byteLength = rawByteLength ?? (serialized === undefined ? undefined : new TextEncoder().encode(serialized).byteLength);
  if (byteLength === undefined || byteLength > MAX_SERIALIZED_BATCH_BYTES) {
    return { ok: false, errors: [{ path: "", keyword: "maxBytes" }] };
  }
  if (!isResultBatchV1(value)) {
    return { ok: false, errors: (validate.errors ?? []).map((error) => ({ path: error.instancePath, keyword: error.keyword })) };
  }
  return { ok: true, value };
}
