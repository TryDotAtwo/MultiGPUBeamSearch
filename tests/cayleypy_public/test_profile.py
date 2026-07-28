from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.cayleypy_public.profile import derive_runtime, serialize_preflight
from tools.kaggle_t4_mlp_profiles import select_profile


REGISTRY = json.loads(
    Path("configs/kaggle_t4_mlp_profiles.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize(
    ("output_dim", "move_count", "expected_parent_batch", "expected_effective"),
    [(1, 24, 2048, 1_056_768), (1, 18, 2730, 1_056_768), (24, 24, 2048, 1_064_960)],
)
def test_derive_runtime_uses_move_count_aware_parent_batching(
    output_dim: int, move_count: int, expected_parent_batch: int, expected_effective: int
) -> None:
    profile = select_profile(REGISTRY, 2**20 + 1, output_dim, move_count)

    plan = derive_runtime(profile, 2**20 + 1, output_dim, move_count)

    assert plan.requested_beam == 2**20 + 1
    assert plan.effective_beam == expected_effective
    assert plan.alignment_delta == expected_effective - (2**20 + 1)
    assert plan.parent_batch == expected_parent_batch
    assert plan.stream3_batch_candidates == expected_parent_batch * move_count * 4
    assert plan.shard_capacity_candidates % 1024 == 0
    assert plan.shard_capacity_candidates >= plan.stream3_batch_candidates
    assert plan.shard_capacity_candidates >= plan.runtime["stream4_batch_candidates"]
    assert plan.shard_capacity_candidates >= plan.runtime["stream4_trigger_candidates"]


def test_derive_runtime_marks_cross_puzzle_18_move_use() -> None:
    profile = select_profile(REGISTRY, 2**20, output_dim=1, move_count=18)

    plan = derive_runtime(profile, 2**20, output_dim=1, move_count=18)

    assert plan.cross_puzzle_profile_note == "measured_24_move_seed"
    assert plan.local_beam == plan.effective_beam // 2


@pytest.mark.parametrize("world_size", [1, 4])
def test_derive_runtime_rejects_non_2xt4_world_sizes(world_size: int) -> None:
    profile = select_profile(REGISTRY, 2**20, output_dim=1, move_count=24)

    with pytest.raises(ValueError, match="world_size"):
        derive_runtime(profile, 2**20, output_dim=1, move_count=24, world_size=world_size)


def test_runtime_mapping_cannot_be_mutated_after_derivation() -> None:
    profile = select_profile(REGISTRY, 2**20, output_dim=1, move_count=24)
    plan = derive_runtime(profile, 2**20, output_dim=1, move_count=24)

    with pytest.raises(TypeError):
        plan.runtime["stream4_batch_candidates"] = 1  # type: ignore[index]
    assert serialize_preflight(plan, profile, 24, 1, 100, 100)["runtime"]["stream4_batch_candidates"] == 98_304
def test_preflight_serialization_records_evidence_budgets_and_tmp_guard() -> None:
    profile = select_profile(REGISTRY, 2**20, output_dim=24, move_count=24)
    plan = derive_runtime(profile, 2**20, output_dim=24, move_count=24)

    payload = serialize_preflight(
        plan,
        profile,
        move_count=24,
        history_ram_bytes=28 * 1024**3,
        history_disk_bytes=32 * 1024**3,
        tmp_free_bytes=40 * 1024**3,
    )

    assert payload["profile_evidence"]["kernel_version"] in {4, 5}
    assert payload["hardware"] == "kaggle_2xt4"
    assert payload["requested_beam"] == plan.requested_beam
    assert payload["effective_beam"] == plan.effective_beam
    assert payload["alignment_delta"] == plan.alignment_delta
    assert payload["move_count"] == 24
    assert payload["capacity_derivation"]["stream3_batch_candidates"] == plan.stream3_batch_candidates
    assert payload["history_budgets"] == {
        "ram_bytes": 28 * 1024**3,
        "disk_bytes": 32 * 1024**3,
    }
    assert payload["tmp_free_bytes"] == 40 * 1024**3


def test_preflight_rejects_unknown_profile_status_and_insufficient_tmp_disk() -> None:
    profile = select_profile(REGISTRY, 2**20, output_dim=1, move_count=24)
    profile["validation_status"] = "unknown"
    plan = derive_runtime(
        {**profile, "validation_status": "measured"}, 2**20, output_dim=1, move_count=24
    )

    with pytest.raises(ValueError, match="validation_status"):
        derive_runtime(profile, 2**20, output_dim=1, move_count=24)
    with pytest.raises(ValueError, match="/tmp"):
        serialize_preflight(
            plan, {**profile, "validation_status": "measured"}, 24, 1, 100, 99
        )
