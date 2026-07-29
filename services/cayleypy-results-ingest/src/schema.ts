import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import schema from "../../../configs/cayleypy_results_schema_v1.json";
import { canonicalJson, computeIdempotency, sha256Hex } from "./ids.js";

export type State = number[];
export type TokenPath = string[];
export type Sha256 = string;
export type ModelClass = "output1" | "output_move_count";

export interface ModelManifestV1 {
  state_len: number; num_classes: number; hd1: number; hd2: number; nrd: number;
  output_dim: number; dtype: "fp16"; normalization: "batchnorm_folded" | "layernorm";
  layout: string; batchnorm?: string; embeddingbag?: string; embedding?: string;
}
export interface ResultEnvelopeV1 {
  schema_version: 1;
  client_submission_id: string;
  run_id: string;
  idempotency_key: Sha256;
  submitted_at: string;
  author: { name: string; kaggle_username?: string; verification: "claimed" };
  kaggle: { owner: string; slug: string; version: number; notebook_sha256: Sha256; run_url?: string };
  competition: string;
  puzzle_type: string;
  puzzle_id: number;
  proof: {
    initial_state: State; central_state: State; generators: Record<string, State>;
    initial_state_sha256: Sha256; central_state_sha256: Sha256; generators_sha256: Sha256; reached_state_sha256: Sha256;
  };
  orientation: {
    search_mode: "off" | "after_original" | "only";
    final_orientation: "original" | "reflected" | "source";
    searched_path?: TokenPath; reflected_source_path?: TokenPath; reflected_source_sha256?: Sha256;
  };
  solution: {
    path: TokenPath; length: number; solved_depth: number; touch_depth?: number; validation: "valid";
    collection_index?: number; collection_status: "first_solution" | "depth_reached" | "capacity_reached" | "not_collected";
  };
  profile: {
    requested_beam: number; effective_beam: number; alignment_delta: number; selected_profile: string; evidence: string;
    profile_evidence_version: number; profile_power: number; model_class: ModelClass; world_size: 2;
  };
  runtime: {
    touch_bfs_radius: number; solution_mode: "first" | "collect"; max_depth: number; max_collected_solutions: number;
    b_micro: number; stream1_concurrency: number; stream3_ring_slots: number; shard_count: number;
    shard_capacity_scale_ppm: number; stream4_batch_candidates: number; stream4_trigger_candidates: number; stream4_active_sort_slots: number;
  };
  model: { filename: string; sha256: Sha256; format: "batchnorm-folded" | "resmlp-layernorm"; manifest: ModelManifestV1 };
  hardware: { platform: string; gpu_names: string[]; accelerator_count: number; world_size: 2 };
  timings: { solve_us: number; wall_us: number };
  solver_commit: string;
}
export interface ResultBatchV1 { schema_version: 1; results: ResultEnvelopeV1[] }
export interface SafeSchemaError { path: string; keyword: string }
export type ValidationResult = { ok: true; value: ResultBatchV1 } | { ok: false; errors: SafeSchemaError[] };

export const MAX_SERIALIZED_BATCH_BYTES = 4 * 1024 * 1024;
export const MAX_SERIALIZED_ENVELOPE_BYTES = 256 * 1024;
// The canonical shared schema composes required/type constraints through allOf.
const ajv = new Ajv2020({ allErrors: true, strict: true, strictRequired: false, strictTypes: false, allowUnionTypes: true });
addFormats(ajv);
const validate = ajv.compile(schema);

function isResultBatchV1(value: unknown): value is ResultBatchV1 { return validate(value); }
function byteLength(value: unknown): number { return new TextEncoder().encode(canonicalJson(value)).byteLength; }
function integrityError(path: string, keyword: string): SafeSchemaError { return { path, keyword }; }
function replay(initial: State, path: TokenPath, generators: Record<string, State>): State | null {
  let state = [...initial];
  for (const token of path) {
    const permutation = generators[token];
    if (!permutation) return null;
    state = permutation.map((source) => state[source]);
  }
  return state;
}

/** Deterministic, server-side semantic and proof checks after JSON-schema validation. */
export async function validateEnvelopeIntegrity(envelope: ResultEnvelopeV1): Promise<SafeSchemaError[]> {
  const errors: SafeSchemaError[] = [];
  const { proof, model, profile, orientation, solution } = envelope;
  const stateLength = model.manifest.state_len;
  if (stateLength < 1 || stateLength > 120) errors.push(integrityError("/model/manifest/state_len", "stateLength"));
  for (const [name, state] of [["initial_state", proof.initial_state], ["central_state", proof.central_state]] as const) {
    if (state.length !== stateLength) errors.push(integrityError(`/proof/${name}`, "stateLength"));
  }
  for (const [name, permutation] of Object.entries(proof.generators)) {
    if (permutation.length !== stateLength || new Set(permutation).size !== stateLength || permutation.some((value) => value < 0 || value >= stateLength)) {
      errors.push(integrityError(`/proof/generators/${name}`, "permutation"));
    }
  }
  const expectedHashes: Array<[string, unknown, string]> = [
    ["/proof/initial_state_sha256", proof.initial_state, proof.initial_state_sha256],
    ["/proof/central_state_sha256", proof.central_state, proof.central_state_sha256],
    ["/proof/generators_sha256", proof.generators, proof.generators_sha256],
  ];
  for (const [path, value, claimed] of expectedHashes) {
    if (await sha256Hex(canonicalJson(value)) !== claimed) errors.push(integrityError(path, "proofHash"));
  }
  if (solution.length !== solution.path.length) errors.push(integrityError("/solution/length", "solutionLength"));
  const reached = replay(proof.initial_state, solution.path, proof.generators);
  if (reached === null) errors.push(integrityError("/solution/path", "unknownMove"));
  else {
    if (await sha256Hex(canonicalJson(reached)) !== proof.reached_state_sha256) errors.push(integrityError("/proof/reached_state_sha256", "proofHash"));
    if (canonicalJson(reached) !== canonicalJson(proof.central_state)) errors.push(integrityError("/solution/path", "replayTarget"));
  }
  const reflected = orientation.final_orientation === "reflected";
  if (reflected) {
    const source = orientation.reflected_source_path;
    if (!source || !orientation.searched_path || !orientation.reflected_source_sha256) errors.push(integrityError("/orientation", "reflectionProvenance"));
    else if (await sha256Hex(source.join(".")) !== orientation.reflected_source_sha256) errors.push(integrityError("/orientation/reflected_source_sha256", "reflectionHash"));
  } else if (orientation.searched_path || orientation.reflected_source_path || orientation.reflected_source_sha256) {
    errors.push(integrityError("/orientation", "reflectionProvenance"));
  }
  const moveCount = Object.keys(proof.generators).length;
  if ((profile.model_class === "output1" && model.manifest.output_dim !== 1) ||
      (profile.model_class === "output_move_count" && model.manifest.output_dim !== moveCount)) {
    errors.push(integrityError("/model/manifest/output_dim", "modelHead"));
  }
  if (await computeIdempotency(envelope) !== envelope.idempotency_key) errors.push(integrityError("/idempotency_key", "idempotency"));
  if (byteLength(envelope) > MAX_SERIALIZED_ENVELOPE_BYTES) errors.push(integrityError("", "maxEnvelopeBytes"));
  return errors;
}

export async function validateBatchIntegrity(results: readonly ResultEnvelopeV1[]): Promise<SafeSchemaError[]> {
  const all = await Promise.all(results.map((envelope) => validateEnvelopeIntegrity(envelope)));
  return all.flatMap((errors, index) => errors.map((error) => ({ path: `/results/${index}${error.path}`, keyword: error.keyword })));
}

export function validateBatch(value: unknown, rawByteLength?: number): ValidationResult {
  let length = rawByteLength;
  if (length === undefined) {
    try { length = byteLength(value); } catch { return { ok: false, errors: [integrityError("", "serialization")] }; }
  }
  if (length === undefined || length > MAX_SERIALIZED_BATCH_BYTES) return { ok: false, errors: [integrityError("", "maxBytes")] };
  if (!isResultBatchV1(value)) return { ok: false, errors: (validate.errors ?? []).map((error) => ({ path: error.instancePath, keyword: error.keyword })) };
  return { ok: true, value };
}