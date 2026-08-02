import json
from pathlib import Path
import pytest
from portable.megaminx_cluster.profile_cache import install_fragment, load_registry

H={"gpu_family":"A100","vram_mib":40960,"sm":80,"world_size":8}
R={"b_micro":1024,"stream1_concurrency":1,"stream3_ring_slots":1,"shard_count":8,"shard_capacity_scale_ppm":1250000,"stream4_batch_candidates":65536,"stream4_trigger_candidates":65536,"stream4_active_sort_slots":1,"final_materialize_chunk_candidates":32768}
def fragment(status="measured", evidence="new"):
 return {"schema_version":1,"profiles":[{"hardware":H,"backend":"mlp","model_class":"output1","min_beam_power":25,"max_beam_power":25,"anchors":{"25":{"status":status,"evidence_id":evidence,"runtime":R}}}]}
def test_install_is_atomic_and_cached_profile_overrides_bundled(tmp_path: Path):
 bundled=tmp_path/"bundled.json"; cache=tmp_path/"cache/registry.json"
 bundled.write_text(json.dumps(fragment(evidence="old")))
 install_fragment(cache,fragment())
 merged=load_registry(bundled,cache)
 assert len(merged["profiles"])==1
 assert merged["profiles"][0]["anchors"]["25"]["evidence_id"]=="new"
def test_unverified_profile_is_never_installed(tmp_path: Path):
 with pytest.raises(ValueError,match="fully measured"):
  install_fragment(tmp_path/"cache.json",fragment("unverified"))
