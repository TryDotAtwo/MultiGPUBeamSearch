from __future__ import annotations

import json
from dataclasses import replace

import pytest

from portable.megaminx_cluster.autotune.evidence import EvidenceStore, SessionIdentity


RUNTIME = {
    "b_micro": 262144,
    "stream1_concurrency": 2,
    "stream3_ring_slots": 2,
    "shard_count": 8,
    "shard_capacity_scale_ppm": 1200000,
    "stream4_batch_candidates": 1048576,
    "stream4_trigger_candidates": 2097152,
    "stream4_active_sort_slots": 2,
    "final_materialize_chunk_candidates": 262144,
}


def identity():
    return SessionIdentity(
        gpu_family="A100", vram_mib=40960, sm=80, world_size=8,
        driver="570.124.06", solver_commit="abc", model_digest="model",
        release_manifest_digest="release", backend="mlp", model_class="output1",
        puzzle_ids=(900, 950, 1000), min_beam=30_000_000,
        time_budget_seconds=21600, bfs_radius=5, move_count=24,
        hash_bytes=16, bfs_hash_budget_bytes=256 * 2**20,
        bfs_cumulative_states=8_308_825,
    )


def test_create_and_exact_resume_identity(tmp_path):
    store = EvidenceStore.create_or_resume(tmp_path, identity())
    assert json.loads((tmp_path / "session.json").read_text())["world_size"] == 8
    assert EvidenceStore.create_or_resume(tmp_path, identity()).identity == identity()
    with pytest.raises(ValueError, match="identity mismatch"):
        EvidenceStore.create_or_resume(tmp_path, replace(identity(), driver="571.0"))


def test_append_only_jsonl_and_corrupt_tail_rejection(tmp_path):
    store = EvidenceStore.create_or_resume(tmp_path, identity())
    store.append_trial({"config_id": "b", "stable": False, "wall_us": 3})
    store.append_trial({"config_id": "a", "stable": True, "wall_us": 2})
    assert len((tmp_path / "autotune_results.jsonl").read_text().splitlines()) == 2
    (tmp_path / "autotune_results.jsonl").open("a", encoding="utf-8").write("{broken\n")
    with pytest.raises(ValueError, match="invalid evidence JSONL"):
        EvidenceStore.create_or_resume(tmp_path, identity())


def test_checkpoint_is_atomic_and_leaderboard_is_deterministic(tmp_path):
    store = EvidenceStore.create_or_resume(tmp_path, identity())
    store.append_trial({"config_id": "b", "stable": False, "wall_us": 1, "peak_vram_mib": 1})
    store.append_trial({"config_id": "a", "stable": True, "wall_us": 2, "peak_vram_mib": 3})
    store.write_checkpoint({"phase": "halving", "anchor_index": 2})
    store.write_leaderboard()
    assert json.loads((tmp_path / "resume.json").read_text())["phase"] == "halving"
    assert not list(tmp_path.glob("*.tmp"))
    lines = (tmp_path / "leaderboard.tsv").read_text().splitlines()
    assert lines[1].startswith("a\t")


def _final_rows(store, anchors=(25, 26)):
    for power in anchors:
        for puzzle in (900, 950, 1000):
            for repetition in range(3):
                store.append_trial({
                    "phase": "final", "profile_power": power, "puzzle_id": puzzle,
                    "repetition": repetition, "config_id": f"p{power}", "beam": 1 << power, "stable": True,
                    "wall_us": 100 + power, "peak_vram_mib": 35000,
                })


def test_registry_is_measured_only_after_all_final_gates(tmp_path):
    store = EvidenceStore.create_or_resume(tmp_path, identity())
    anchors = {
        25: {"runtime": RUNTIME, "evidence_id": "ev25", "config_id": "p25", "beam": 1 << 25},
        26: {"runtime": RUNTIME, "evidence_id": "ev26", "config_id": "p26", "beam": 1 << 26},
    }
    partial = store.emit_registry_fragment(anchors)
    assert partial["profiles"][0]["anchors"]["25"]["status"] == "unverified"
    _final_rows(store)
    measured = store.emit_registry_fragment(anchors)
    anchor = measured["profiles"][0]["anchors"]["25"]
    assert anchor["status"] == "measured"
    assert anchor["bfs"]["radius"] == 5
    assert anchor["bfs"]["hash_bytes"] == 16
    assert json.loads((tmp_path / "registry.fragment.json").read_text()) == measured


def test_one_failed_final_row_keeps_anchor_unverified(tmp_path):
    store = EvidenceStore.create_or_resume(tmp_path, identity())
    _final_rows(store, anchors=(25,))
    store.append_trial({
        "phase": "final", "profile_power": 25, "puzzle_id": 900,
        "repetition": 3, "config_id": "p25", "beam": 1 << 25, "stable": False, "wall_us": 1,
    })
    fragment = store.emit_registry_fragment({25: {
        "runtime": RUNTIME, "evidence_id": "ev25", "config_id": "p25", "beam": 1 << 25,
    }})
    assert fragment["profiles"][0]["anchors"]["25"]["status"] == "unverified"

def test_emitted_fragment_matches_profile_schema(tmp_path):
    from pathlib import Path
    from jsonschema import Draft202012Validator

    store = EvidenceStore.create_or_resume(tmp_path, identity())
    fragment = store.emit_registry_fragment({25: {
        "runtime": RUNTIME, "evidence_id": "ev25", "config_id": "p25", "beam": 1 << 25,
    }})
    schema = json.loads(Path("portable/megaminx_cluster/profiles/schema.json").read_text())
    Draft202012Validator(schema).validate(fragment)


def test_failed_other_config_does_not_poison_measured_anchor(tmp_path):
    store = EvidenceStore.create_or_resume(tmp_path, identity())
    _final_rows(store, anchors=(25,))
    store.append_trial({
        "phase": "final", "profile_power": 25, "beam": 1 << 25,
        "puzzle_id": 900, "repetition": 0, "config_id": "other",
        "stable": False, "wall_us": 1,
    })
    fragment = store.emit_registry_fragment({25: {
        "runtime": RUNTIME, "evidence_id": "ev25", "config_id": "p25", "beam": 1 << 25,
    }})
    assert fragment["profiles"][0]["anchors"]["25"]["status"] == "measured"
