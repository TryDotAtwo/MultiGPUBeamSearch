from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools.kaggle_t4_mlp_profiles import (
    align_beam,
    round_half_up_log2,
    select_profile,
)


def make_profiles() -> dict:
    runtime = {
        "b_micro": 49152,
        "stream1_concurrency": 4,
        "stream3_ring_slots": 4,
        "shard_count": 8,
        "shard_capacity_scale_ppm": 1050000,
        "stream4_batch_candidates": 98304,
        "stream4_trigger_candidates": 98304,
        "stream4_active_sort_slots": 4,
    }
    anchors = {
        str(power): {
            "validation_status": "seed",
            "runtime": runtime,
        }
        for power in range(16, 26)
    }
    return {
        "schema_version": 1,
        "hardware": "kaggle_2xt4",
        "profiles": {
            "output1": anchors,
            "output_move_count": anchors,
        },
    }


def test_power_of_two_and_clamps() -> None:
    assert round_half_up_log2(2**20) == 20
    assert round_half_up_log2(1) == 16
    assert round_half_up_log2(2**30) == 25


def test_exact_geometric_midpoint_rounds_up() -> None:
    integer_above_midpoint = math.ceil(2**20 * 2**0.5)
    assert round_half_up_log2(integer_above_midpoint) == 21


def test_non_power_beam_selects_nearest_log_anchor() -> None:
    assert round_half_up_log2(24_000_000) == 25


def test_profile_selection_does_not_replace_requested_beam() -> None:
    selected = select_profile(make_profiles(), 24_000_000, output_dim=1, move_count=24)
    assert selected["requested_beam"] == 24_000_000
    assert selected["profile_power"] == 25
    assert selected["model_class"] == "output1"


def test_alignment_is_the_only_beam_adjustment() -> None:
    assert align_beam(1_000_003, 2, 8, 1024) == 1_015_808


def test_output_move_count_requires_exact_match() -> None:
    with pytest.raises(ValueError, match="output_dim"):
        select_profile(make_profiles(), 2**20, output_dim=12, move_count=24)


def test_output_move_count_selects_separate_profile_family() -> None:
    selected = select_profile(make_profiles(), 2**20, output_dim=24, move_count=24)
    assert selected["model_class"] == "output_move_count"


@pytest.mark.parametrize("beam_width", [0, -1])
def test_beam_must_be_positive(beam_width: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        round_half_up_log2(beam_width)


def test_registry_hardware_must_be_kaggle_2xt4() -> None:
    profiles = make_profiles()
    profiles["hardware"] = "h100"
    with pytest.raises(ValueError, match="kaggle_2xt4"):
        select_profile(profiles, 2**20, output_dim=1, move_count=24)


def test_profile_status_must_be_explicit() -> None:
    profiles = make_profiles()
    profiles["profiles"]["output1"]["20"]["validation_status"] = "unknown"
    with pytest.raises(ValueError, match="validation_status"):
        select_profile(profiles, 2**20, output_dim=1, move_count=24)


def test_checked_in_registry_has_twenty_measured_anchors() -> None:
    registry = json.loads(
        Path("configs/kaggle_t4_mlp_profiles.json").read_text(encoding="utf-8")
    )
    measured = []
    for model_class in ("output1", "output_move_count"):
        assert set(registry["profiles"][model_class]) == {
            str(power) for power in range(16, 26)
        }
        for power, profile in registry["profiles"][model_class].items():
            assert profile["validation_status"] == "measured"
            assert profile["evidence"]["kernel_version"] in {4, 5}
            assert profile["evidence"]["depth"] == 8
            assert profile["evidence"]["depth_sec"] > 0
            measured.append((model_class, power))
    assert len(measured) == 20