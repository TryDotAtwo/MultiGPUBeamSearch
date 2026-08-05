import { describe, expect, test } from "vitest";
import v1Golden from "../../../configs/cayleypy_results_v1_golden.json";
import v2Golden from "../../../configs/cayleypy_results_v2_golden.json";
import v2ExamplePayload from "../../../configs/cayleypy_results_v2_example_payload.json";
import { canonicalJson, sha256Hex } from "../src/ids.js";
import { validateBatchV2, validateEnvelopeIntegrityV2, type ResultEnvelopeV2 } from "../src/schema-v2.js";

async function validV2(): Promise<ResultEnvelopeV2> {
  const base = structuredClone(v1Golden.cases[0].envelope) as any;
  delete base.kaggle; delete base.solver_commit;
  base.schema_version = 2;
  base.author = { name: "portable-cluster-user", verification: "claimed" };
  base.provenance = { platform: "slurm", cluster_name: "basis", slurm_job_id: "32633", slurm_array_task_id: null, run_id: base.run_id, release_tag: "megaminx-native-v1", release_asset: "megaminx-sm90-linux-x86_64.tar.zst", release_manifest_sha256: "a".repeat(64), solver_commit: "b".repeat(40) };
  base.hardware = { platform: "slurm", gpu_names: Array(4).fill("NVIDIA H100 80GB HBM3"), accelerator_count: 4, world_size: 4, native_sm: 90, vram_mib_per_gpu: 81559, driver_version: "570.133.20" };
  base.profile = { requested_beam: 1_000_000_000, effective_beam: 1_000_013_824, alignment_delta: 13_824, profile_power: 30, profile_anchor_beam: 1_073_741_824, profile_status: "measured", profile_evidence_id: "h100x4-megaminx-p30-v1", gpu_family: "H100", vram_mib: 81559, native_sm: 90, world_size: 4, backend: "mlp", model_class: "output_move_count" };
  base.orientation.search_mode = "off";
  base.idempotency_key = "0".repeat(64);
  const { client_submission_id: _c, run_id: _r, idempotency_key: _i, submitted_at: _s, ...semantic } = base;
  base.idempotency_key = await sha256Hex(canonicalJson(semantic));
  return base;
}
const rehash = async (envelope: ResultEnvelopeV2) => { envelope.idempotency_key = "0".repeat(64); const { client_submission_id: _c, run_id: _r, idempotency_key: _i, submitted_at: _s, ...semantic } = envelope; envelope.idempotency_key = await sha256Hex(canonicalJson(semantic)); };

describe("canonical CayleyPy SLURM result v2", () => {
  test("accepts H100x4 native-SM without Kaggle provenance", async () => { const envelope = await validV2(); expect(validateBatchV2({ schema_version: 2, results: [envelope] })).toMatchObject({ ok: true }); expect(await validateEnvelopeIntegrityV2(envelope)).toEqual([]); expect((envelope as any).kaggle).toBeUndefined(); });
  test.each([[7,80,"NVIDIA A100-SXM4-80GB","A100"],[4,90,"NVIDIA H100 80GB HBM3","H100"],[1,89,"NVIDIA L4","L4"],[4,89,"NVIDIA L4","L4"],[4,120,"NVIDIA GeForce RTX 5090","Blackwell"]])("accepts count=%s sm=%s", async (count, sm, name, family) => { const e=await validV2(); e.hardware.gpu_names=Array(count).fill(name); e.hardware.accelerator_count=count; e.hardware.world_size=count; e.hardware.native_sm=sm as any; e.profile.world_size=count; e.profile.native_sm=sm as any; e.profile.gpu_family=family; await rehash(e); expect(validateBatchV2({schema_version:2,results:[e]})).toMatchObject({ok:true}); expect(await validateEnvelopeIntegrityV2(e)).toEqual([]); });
  test.each([0,17])("rejects world_size=%s", async (worldSize) => { const e=await validV2(); e.hardware.world_size=worldSize; expect(validateBatchV2({schema_version:2,results:[e]})).toMatchObject({ok:false}); });
  test("rejects cardinality, mixed GPU, and cross-hardware profile", async () => { const e=await validV2(); e.hardware.accelerator_count=3; e.hardware.gpu_names[1]="NVIDIA A100-SXM4-80GB"; e.profile.native_sm=80; const errors=await validateEnvelopeIntegrityV2(e); expect(errors).toContainEqual({path:"/hardware/accelerator_count",keyword:"hardwareCardinality"}); expect(errors).toContainEqual({path:"/hardware/gpu_names",keyword:"mixedHardware"}); expect(errors).toContainEqual({path:"/profile/native_sm",keyword:"profileHardware"}); });
  test("rejects unverified profile and beam alignment mismatch", async () => { const e=await validV2(); (e.profile as any).profile_status="unverified"; e.profile.alignment_delta+=1; expect(validateBatchV2({schema_version:2,results:[e]})).toMatchObject({ok:false}); expect(await validateEnvelopeIntegrityV2(e)).toContainEqual({path:"/profile/alignment_delta",keyword:"beamAlignment"}); });
  test("rejects Kaggle, private provenance, and unknown fields", async () => { for (const mutate of [(x:any)=>x.kaggle={owner:"fake"},(x:any)=>x.provenance.hostname="private",(x:any)=>x.provenance.absolute_path="/private/model.pt"]) { const e:any=await validV2(); mutate(e); expect(validateBatchV2({schema_version:2,results:[e]})).toMatchObject({ok:false}); } });
  test("accepts canonical original and reflected golden replay chains", async () => { expect(v2Golden.cases).toHaveLength(2); for (const item of v2Golden.cases) { const envelope=structuredClone(item.envelope) as unknown as ResultEnvelopeV2; expect(validateBatchV2({schema_version:2,results:[envelope]})).toMatchObject({ok:true}); expect(await validateEnvelopeIntegrityV2(envelope)).toEqual([]); } });  test("rejects wrong backend and cross-world profile", async () => { const e=await validV2(); e.profile.backend="piece_transformer"; e.profile.world_size=7; const errors=await validateEnvelopeIntegrityV2(e); expect(errors).toContainEqual({path:"/profile/backend",keyword:"profileBackend"}); expect(errors).toContainEqual({path:"/profile/native_sm",keyword:"profileHardware"}); });
  test("ships a directly POSTable canonical example payload", async () => {
    expect(validateBatchV2(v2ExamplePayload)).toMatchObject({ ok: true });
    expect(await validateEnvelopeIntegrityV2(v2ExamplePayload.results[0] as ResultEnvelopeV2)).toEqual([]);
  });
  test("rejects paths, logs, tokens, weights, and arbitrary markdown", async () => { for (const mutate of [(e:any)=>e.model.filename="/private/model.pt",(e:any)=>e.logs=["secret"],(e:any)=>e.token="ghp_secret",(e:any)=>e.weights="blob",(e:any)=>e.markdown="# arbitrary"]) { const e:any=await validV2(); mutate(e); expect(validateBatchV2({schema_version:2,results:[e]})).toMatchObject({ok:false}); } });});