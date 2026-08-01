import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import schema from "../../../configs/cayleypy_results_schema_v2.json";
import { canonicalJson, computeIdempotency } from "./ids.js";
import { validateEnvelopeIntegrity, MAX_SERIALIZED_ENVELOPE_BYTES, type ModelManifestV1, type ResultEnvelopeV1, type SafeSchemaError, type State, type TokenPath, type Sha256, type ModelClass } from "./schema.js";

export type NativeSm = 75 | 80 | 86 | 89 | 90 | 120;
export interface ResultEnvelopeV2 {
  schema_version: 2; client_submission_id: string; run_id: string; idempotency_key: Sha256; submitted_at: string;
  author: { name: string; verification: "claimed" };
  provenance: { platform: "slurm"; cluster_name: string; slurm_job_id: string; slurm_array_task_id: string | null; run_id: string; release_tag: string; release_asset: string; release_manifest_sha256: Sha256; solver_commit: string };
  competition: string; puzzle_type: string; puzzle_id: number;
  proof: { initial_state: State; central_state: State; generators: Record<string, State>; initial_state_sha256: Sha256; central_state_sha256: Sha256; generators_sha256: Sha256; reached_state_sha256: Sha256 };
  orientation: { search_mode: "off" | "after" | "only"; final_orientation: "original" | "reflected" | "source"; searched_path?: TokenPath; reflected_source_path?: TokenPath; reflected_source_sha256?: Sha256 };
  solution: { path: TokenPath; length: number; solved_depth: number; touch_depth?: number; validation: "valid"; collection_index?: number; collection_status: "first_solution" | "depth_reached" | "capacity_reached" | "not_collected" };
  profile: { requested_beam: number; effective_beam: number; alignment_delta: number; profile_power: number; profile_anchor_beam: number; profile_status: "measured" | "bounded_from_measured"; profile_evidence_id: string; gpu_family: string; vram_mib: number; native_sm: NativeSm; world_size: number; backend: "mlp" | "piece_transformer"; model_class: ModelClass };
  runtime: ResultEnvelopeV1["runtime"];
  model: { filename: string; sha256: Sha256; format: ResultEnvelopeV1["model"]["format"]; manifest: ModelManifestV1 };
  hardware: { platform: "slurm"; gpu_names: string[]; accelerator_count: number; world_size: number; native_sm: NativeSm; vram_mib_per_gpu: number; driver_version: string };
  timings: ResultEnvelopeV1["timings"];
}
export interface ResultBatchV2 { schema_version: 2; results: ResultEnvelopeV2[] }
export type ValidationResultV2 = { ok: true; value: ResultBatchV2 } | { ok: false; errors: SafeSchemaError[] };

const ajv = new Ajv2020({ allErrors: true, strict: true, strictRequired: false, strictTypes: false, allowUnionTypes: true });
addFormats(ajv);
const validate = ajv.compile(schema);
const byteLength = (value: unknown): number => new TextEncoder().encode(canonicalJson(value)).byteLength;
const error = (path: string, keyword: string): SafeSchemaError => ({ path, keyword });

export function validateBatchV2(value: unknown, rawByteLength?: number, maxBytes = 64 * 1024 * 1024): ValidationResultV2 {
  let length = rawByteLength;
  if (length === undefined) { try { length = byteLength(value); } catch { return { ok: false, errors: [error("", "serialization")] }; } }
  if (length === undefined || length > maxBytes) return { ok: false, errors: [error("", "maxBytes")] };
  if (!validate(value)) return { ok: false, errors: (validate.errors ?? []).map((item) => ({ path: item.instancePath, keyword: item.keyword })) };
  return { ok: true, value: value as unknown as ResultBatchV2 };
}

function inferredHardware(name: string): { sm: NativeSm; family: string } | null {
  const upper = name.toUpperCase();
  if (upper.includes("T4")) return { sm: 75, family: "T4" };
  if (upper.includes("A100")) return { sm: 80, family: "A100" };
  if (upper.includes("RTX 30") || upper.includes("A10")) return { sm: 86, family: upper.includes("A10") ? "A10" : "Ampere" };
  if (upper.includes("L4")) return { sm: 89, family: "L4" };
  if (upper.includes("RTX 40") || upper.includes("L40")) return { sm: 89, family: "Ada" };
  if (upper.includes("H100")) return { sm: 90, family: "H100" };
  if (upper.includes("BLACKWELL") || upper.includes("RTX 50")) return { sm: 120, family: "Blackwell" };
  return null;
}

function v1ReplayView(envelope: ResultEnvelopeV2): ResultEnvelopeV1 {
  const view = {
    ...envelope, schema_version: 1 as const,
    kaggle: { owner: "slurm", slug: "v2-replay", version: 1, notebook_sha256: "0".repeat(64) },
    solver_commit: envelope.provenance.solver_commit,
    orientation: { ...envelope.orientation, search_mode: envelope.orientation.search_mode === "after" ? "after_original" as const : envelope.orientation.search_mode },
    profile: { requested_beam: envelope.profile.requested_beam, effective_beam: envelope.profile.effective_beam, alignment_delta: envelope.profile.alignment_delta, selected_profile: envelope.profile.profile_evidence_id, evidence: envelope.profile.profile_status, profile_evidence_version: 1, profile_power: envelope.profile.profile_power, model_class: envelope.profile.model_class, world_size: 2 as const },
    hardware: { platform: "slurm", gpu_names: envelope.hardware.gpu_names, accelerator_count: envelope.hardware.accelerator_count, world_size: 2 as const },
  } as unknown as ResultEnvelopeV1;
  return view;
}

export async function validateEnvelopeIntegrityV2(envelope: ResultEnvelopeV2): Promise<SafeSchemaError[]> {
  const errors: SafeSchemaError[] = [];
  const replayView = v1ReplayView(envelope);
  replayView.idempotency_key = await computeIdempotency(replayView);
  errors.push(...(await validateEnvelopeIntegrity(replayView)).filter((item) => item.keyword !== "idempotency"));
  const { hardware, profile, provenance } = envelope;
  if (hardware.gpu_names.length !== hardware.accelerator_count || hardware.world_size !== hardware.accelerator_count) errors.push(error("/hardware/accelerator_count", "hardwareCardinality"));
  const uniqueNames = new Set(hardware.gpu_names);
  if (uniqueNames.size !== 1) errors.push(error("/hardware/gpu_names", "mixedHardware"));
  const inferred = uniqueNames.size === 1 ? inferredHardware(hardware.gpu_names[0] ?? "") : null;
  if (inferred === null || inferred.sm !== hardware.native_sm) errors.push(error("/hardware/native_sm", "nativeSm"));
  if (profile.effective_beam < profile.requested_beam || profile.alignment_delta !== profile.effective_beam - profile.requested_beam) errors.push(error("/profile/alignment_delta", "beamAlignment"));
  if (profile.profile_anchor_beam !== 2 ** profile.profile_power) errors.push(error("/profile/profile_anchor_beam", "profileAnchor"));
  if (profile.native_sm !== hardware.native_sm || profile.world_size !== hardware.world_size || profile.vram_mib !== hardware.vram_mib_per_gpu) errors.push(error("/profile/native_sm", "profileHardware"));
  if (inferred !== null && profile.gpu_family !== inferred.family) errors.push(error("/profile/gpu_family", "profileHardware"));
  const expectedBackend = envelope.model.format === "piece-transformer" ? "piece_transformer" : "mlp";
  if (profile.backend !== expectedBackend) errors.push(error("/profile/backend", "profileBackend"));
  if (provenance.run_id !== envelope.run_id) errors.push(error("/provenance/run_id", "runProvenance"));
  if (await computeIdempotency(envelope) !== envelope.idempotency_key) errors.push(error("/idempotency_key", "idempotency"));
  if (byteLength(envelope) > MAX_SERIALIZED_ENVELOPE_BYTES) errors.push(error("", "maxEnvelopeBytes"));
  return errors;
}

export async function validateBatchIntegrityV2(results: readonly ResultEnvelopeV2[]): Promise<SafeSchemaError[]> {
  const all = await Promise.all(results.map(validateEnvelopeIntegrityV2));
  return all.flatMap((items, index) => items.map((item) => ({ path: `/results/${index}${item.path}`, keyword: item.keyword })));
}