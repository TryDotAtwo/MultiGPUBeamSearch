import json
from pathlib import Path

import yaml

from tools.measure_megaminx_cluster_profiles import select_winners


ROOT = Path(__file__).resolve().parents[2]


def test_release_workflow_has_exact_six_native_targets_and_prerelease_gate():
    text = (ROOT / ".github/workflows/megaminx-native-release.yml").read_text()
    workflow = yaml.safe_load(text)
    assert workflow["jobs"]["build"]["strategy"]["matrix"]["sm"] == [75, 80, 86, 89, 90, 120]
    assert "cuda-12.8" in text
    assert "--prerelease" in text
    assert "--draft=false" not in text
    assert "compute_" not in text and "PTX" not in text
    assert "tools.check_megaminx_native_archive" in text


def candidate(power, median_bias, digest="exact", status="measured"):
    return {
        "hardware": {"gpu_family": "H100", "vram_mib": 81559, "sm": 90, "world_size": 4},
        "backend": "mlp", "model_class": "output_move_count", "beam_power": power,
        "status": status, "evidence_id": f"h100x4-p{power}-{median_bias}",
        "correctness_digest": digest, "expected_correctness_digest": "exact",
        "timings_ms": [10 + median_bias, 11 + median_bias, 12 + median_bias],
        "runtime": {"b_micro": 2048, "stream1_concurrency": 4, "stream3_ring_slots": 4, "shard_count": 4, "shard_capacity_scale_ppm": 1050000, "stream4_batch_candidates": 98304, "stream4_trigger_candidates": 98304, "stream4_active_sort_slots": 4, "final_materialize_chunk_candidates": 88064},
    }


def test_profile_sweep_selects_fastest_repeated_exact_candidate_per_power():
    winners = select_winners([candidate(29, 5), candidate(29, 0), candidate(30, 2)])
    assert [item["beam_power"] for item in winners] == [29, 30]
    assert winners[0]["evidence_id"] == "h100x4-p29-0"
    assert winners[0]["median_ms"] == 11


def test_profile_sweep_rejects_wrong_digest_and_insufficient_repeats():
    bad_digest = candidate(29, 0, digest="wrong")
    too_short = candidate(29, 0)
    too_short["timings_ms"] = [10, 11]
    assert select_winners([bad_digest, too_short]) == []


def test_profile_sweep_is_deterministic_under_input_reordering():
    items = [candidate(30, 2), candidate(29, 0), candidate(29, 5)]
    assert json.dumps(select_winners(items), sort_keys=True) == json.dumps(select_winners(list(reversed(items))), sort_keys=True)

def test_cluster_branch_push_publishes_stable_latest_download_assets():
    text = (ROOT / ".github/workflows/megaminx-native-release.yml").read_text()
    assert "codex/megaminx-native-cluster-release" in text
    assert "megaminx-native-cluster-latest" in text
    assert "gh release upload" in text
    assert "--clobber" in text

def test_release_build_bootstraps_cuda_on_github_hosted_linux():
    text = (ROOT / ".github/workflows/megaminx-native-release.yml").read_text()
    assert "runs-on: ubuntu-22.04" in text
    assert "cuda-keyring" in text
    assert "cuda-nvcc-12-8" in text
    assert "libnccl-dev" in text
    assert "NVIDIA/cutlass" in text

def test_release_tag_is_bound_in_build_and_publish_jobs():
    workflow = yaml.safe_load((ROOT / ".github/workflows/megaminx-native-release.yml").read_text())
    assert "PRERELEASE_TAG" in workflow["jobs"]["build"]["env"]
    publish = next(step for step in workflow["jobs"]["prerelease"]["steps"] if step.get("name") == "Publish prerelease assets only")
    assert "PRERELEASE_TAG" in publish["env"]

def test_release_runner_installs_archive_python_dependency():
    text = (ROOT / ".github/workflows/megaminx-native-release.yml").read_text()
    assert "python3 -m pip install zstandard==" in text

def test_release_invokes_package_tools_as_modules():
    text = (ROOT / ".github/workflows/megaminx-native-release.yml").read_text()
    assert "python3 -m tools.build_megaminx_native_release" in text
    assert "python3 -m tools.check_megaminx_native_archive" in text
