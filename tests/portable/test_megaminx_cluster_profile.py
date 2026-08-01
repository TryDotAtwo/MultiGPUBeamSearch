import math

import pytest

from portable.megaminx_cluster.profile import (
    HardwareKey,
    derive_capacities,
    round_half_up_log2,
    select_profile,
)


def registry():
    runtime = {
        "b_micro": 49152,
        "stream1_concurrency": 4,
        "stream3_ring_slots": 4,
        "shard_count": 8,
        "shard_capacity_scale_ppm": 1250000,
        "stream4_batch_candidates": 98304,
        "stream4_trigger_candidates": 196608,
        "stream4_active_sort_slots": 4,
        "final_materialize_chunk_candidates": 98304,
    }
    return {
        "schema_version": 1,
        "profiles": [
            {
                "hardware": {"gpu_family": "A100", "vram_mib": 40960, "sm": 80, "world_size": 4},
                "backend": "mlp",
                "model_class": "output1",
                "min_beam_power": 28,
                "max_beam_power": 30,
                "anchors": {
                    "28": {"status": "measured", "evidence_id": "a100x4-p28", "runtime": runtime},
                    "29": {"status": "bounded_from_measured", "evidence_id": "a100x4-p29", "runtime": runtime},
                    "30": {"status": "unverified", "evidence_id": "a100x4-p30", "runtime": runtime},
                },
            }
        ],
    }


@pytest.mark.parametrize(
    ("beam", "power"),
    [(2**28, 28), (int(2 ** 28.49), 28), (math.ceil(2 ** 28.5), 29), (2**29, 29)],
)
def test_round_half_up_log2(beam, power):
    assert round_half_up_log2(beam) == power


def test_selects_exact_hardware_and_preserves_requested_beam():
    requested = 2**28 + 123
    selected = select_profile(
        registry(), HardwareKey("A100", 40960, 80, 4), requested, "mlp", "output1"
    )
    assert selected.requested_beam == requested
    assert selected.profile_power == 28
    assert selected.evidence_id == "a100x4-p28"


@pytest.mark.parametrize(
    "hardware",
    [
        HardwareKey("H100", 81559, 90, 4),
        HardwareKey("A100", 40960, 80, 2),
        HardwareKey("A100", 81920, 80, 4),
    ],
)
def test_rejects_cross_hardware_profiles(hardware):
    with pytest.raises(ValueError, match="no exact profile registry"):
        select_profile(registry(), hardware, 2**28, "mlp", "output1")


def test_rejects_wrong_backend_and_out_of_range():
    hardware = HardwareKey("A100", 40960, 80, 4)
    with pytest.raises(ValueError, match="no exact profile registry"):
        select_profile(registry(), hardware, 2**28, "piece_transformer", "output1")
    with pytest.raises(ValueError, match="supported beam powers are 28..30"):
        select_profile(registry(), hardware, 2**27, "mlp", "output1")


def test_rejects_unverified_anchor_without_fallback():
    with pytest.raises(ValueError, match="profile status unverified is not runnable"):
        select_profile(
            registry(), HardwareKey("A100", 40960, 80, 4), 2**30, "mlp", "output1"
        )


def test_derives_actual_beam_capacities():
    requested = 2**28 + 123
    selected = select_profile(
        registry(), HardwareKey("A100", 40960, 80, 4), requested, "mlp", "output1"
    )
    plan = derive_capacities(selected, requested, move_count=24, output_dim=1)
    assert plan.requested_beam == requested
    assert plan.effective_beam % (4 * 8 * 1024) == 0
    assert plan.effective_beam >= requested
    assert plan.alignment_delta == plan.effective_beam - requested
    assert plan.parent_batch == 49152 // 24
    assert plan.stream3_batch_candidates == (49152 // 24) * 24 * 4
    assert plan.shard_capacity_candidates >= plan.stream3_batch_candidates
    assert plan.shard_capacity_candidates >= 196608


def test_output_dimension_must_match_model_class():
    selected = select_profile(
        registry(), HardwareKey("A100", 40960, 80, 4), 2**28, "mlp", "output1"
    )
    with pytest.raises(ValueError, match="output_dim"):
        derive_capacities(selected, 2**28, move_count=24, output_dim=24)
