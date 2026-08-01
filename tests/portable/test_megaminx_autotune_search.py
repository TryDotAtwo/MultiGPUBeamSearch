from __future__ import annotations

import pytest

from portable.megaminx_cluster.autotune.contracts import PuzzleContract
from portable.megaminx_cluster.autotune.search_space import (
    beam_anchors,
    derive_bfs_boundary,
    refine_max_stable,
)


def test_megaminx_hash128_budget_selects_radius_five():
    boundary = derive_bfs_boundary(PuzzleContract(24, 16, 12), 256 * 2**20)
    assert boundary.radius == 5
    assert boundary.cumulative_states == 8_308_825
    assert boundary.raw_bytes == 132_941_200
    assert boundary.budget_bytes == 256 * 2**20


def test_hash_width_is_part_of_the_puzzle_contract():
    boundary = derive_bfs_boundary(PuzzleContract(24, 8, 12), 256 * 2**20)
    assert boundary.radius == 5
    assert boundary.raw_bytes == 66_470_600


def test_exact_budget_includes_the_last_radius():
    boundary = derive_bfs_boundary(PuzzleContract(3, 16, 12), 40 * 16)
    assert (boundary.radius, boundary.cumulative_states, boundary.raw_bytes) == (3, 40, 640)


@pytest.mark.parametrize(
    "contract,budget",
    [
        (PuzzleContract(0, 16, 12), 1024),
        (PuzzleContract(24, 0, 12), 1024),
        (PuzzleContract(24, 16, -1), 1024),
        (PuzzleContract(24, 16, 12), 0),
    ],
)
def test_invalid_bfs_inputs_fail_closed(contract, budget):
    with pytest.raises(ValueError):
        derive_bfs_boundary(contract, budget)


def test_budget_must_fit_radius_zero_hash():
    with pytest.raises(ValueError, match="radius zero"):
        derive_bfs_boundary(PuzzleContract(24, 16, 12), 15)


def test_schema_radius_cap_is_respected():
    boundary = derive_bfs_boundary(PuzzleContract(2, 16, 3), 10**9)
    assert boundary.radius == 3
    assert boundary.cumulative_states == 15


def test_checked_u64_overflow_fails_closed():
    with pytest.raises(OverflowError):
        derive_bfs_boundary(PuzzleContract(2**63, 1, 2), 2**64 - 1)


def test_beam_anchors_include_exact_endpoints_and_half_up_powers():
    assert beam_anchors(1_000_000_000, 30_000_000) == (
        30_000_000,
        33_554_432,
        67_108_864,
        134_217_728,
        268_435_456,
        536_870_912,
        1_000_000_000,
    )


def test_beam_anchors_reject_reversed_range():
    with pytest.raises(ValueError):
        beam_anchors(29_999_999, 30_000_000)


def test_refine_max_stable_keeps_all_attempts_and_binary_refines():
    result = refine_max_stable(lambda beam: beam <= 100, min_beam=30, max_beam_limit=200)
    assert result.maximum_stable == 100
    assert result.attempts[:3] == ((30, True), (60, True), (120, False))
    assert result.attempts[-1] == (100, True)
    assert len({beam for beam, _ in result.attempts}) == len(result.attempts)


def test_refine_max_stable_rejects_unstable_minimum():
    with pytest.raises(RuntimeError, match="minimum beam"):
        refine_max_stable(lambda beam: False, min_beam=30, max_beam_limit=200)
