from __future__ import annotations

import pytest

from portable.megaminx_cluster.autotune.contracts import TrialScore
from portable.megaminx_cluster.autotune.search_space import (
    candidate_grid,
    retain_round,
    round_schedule,
)
from portable.megaminx_cluster.profile import RUNTIME_KEYS


SEED = {
    "b_micro": 262144,
    "stream1_concurrency": 2,
    "stream3_ring_slots": 2,
    "shard_count": 8,
    "shard_capacity_scale_ppm": 1200000,
    "stream4_batch_candidates": 1048576,
    "stream4_trigger_candidates": 2097152,
    "stream4_active_sort_slots": 2,
    "final_materialize_chunk_candidates": 65536,
}


def test_candidate_grid_is_deterministic_and_uses_only_runtime_contract_keys():
    first = candidate_grid(SEED)
    second = candidate_grid(dict(reversed(tuple(SEED.items()))))
    assert first == second
    assert len(first) == len({item.config_id for item in first})
    assert all(set(item.runtime) == RUNTIME_KEYS for item in first)
    assert first[0].runtime == SEED


def test_candidate_ids_are_content_derived():
    candidates = candidate_grid(SEED)
    assert candidates[0].config_id != candidates[1].config_id
    assert dict(candidates[0].runtime) != dict(candidates[1].runtime)


def test_candidate_grid_rejects_unknown_or_missing_runtime_keys():
    with pytest.raises(ValueError, match="runtime keys"):
        candidate_grid({**SEED, "mystery": 1})
    missing = dict(SEED)
    missing.pop("b_micro")
    with pytest.raises(ValueError, match="runtime keys"):
        candidate_grid(missing)


def test_round_schedule_expands_coverage_and_finishes_with_three_repetitions():
    rounds = round_schedule((900, 901, 902))
    assert rounds[0].warmups == 0
    assert rounds[0].repetitions == 1
    assert rounds[0].puzzle_ids == (900,)
    assert rounds[-1].warmups == 0
    assert rounds[-1].repetitions == 3
    assert rounds[-1].puzzle_ids == (900, 901, 902)


def test_round_schedule_requires_three_distinct_puzzles():
    with pytest.raises(ValueError):
        round_schedule((900, 900, 902))


def test_retain_round_prefers_stability_then_time_memory_and_id():
    scores = (
        TrialScore("failed-fast", False, 1, 1),
        TrialScore("c", True, 100, 900),
        TrialScore("b", True, 100, 800),
        TrialScore("a", True, 100, 800),
        TrialScore("fast", True, 90, 1000),
    )
    retained = retain_round(scores, fraction=0.5)
    assert tuple(item.config_id for item in retained) == ("fast", "a", "b")


def test_retain_round_keeps_at_least_one_and_rejects_fraction():
    score = TrialScore("only", False, None, None)
    assert retain_round((score,), 0.25) == (score,)
    with pytest.raises(ValueError):
        retain_round((score,), 0)


def test_retain_round_chooses_lower_memory_inside_three_percent_speed_band():
    scores = (
        TrialScore("fast-heavy", True, 100, 34000),
        TrialScore("near-fast-light", True, 102, 24000),
        TrialScore("too-slow-light", True, 104, 10000),
    )
    assert retain_round(scores, 1 / 3)[0].config_id == "near-fast-light"