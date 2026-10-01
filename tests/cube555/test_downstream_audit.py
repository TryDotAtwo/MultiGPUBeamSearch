from pathlib import Path

import pytest

from tools.cube555.downstream_audit import downstream_plan, saturated_layers, paired_comparison


@pytest.mark.parametrize('before,candidate,after,resolved', [
    (100, 99, 103, False), (100, 95, 101, True),
    (100, 100, 100, False), (100, 105, 100, False),
])
def test_surrounding_baseline_drift_must_be_smaller_than_gain(before, candidate, after, resolved):
    rows = [dict(status='complete', median_seconds=t) for t in (before, candidate, after)]
    assert paired_comparison(*rows)['resolved'] is resolved


def test_unstable_block_cannot_select_winner():
    rows = [dict(status=s, median_seconds=t) for s, t in
            [('complete', 100), ('unstable', 80), ('complete', 100)]]
    assert not paired_comparison(*rows)['resolved']
from tools.cube555.run import runtime_plan


@pytest.mark.parametrize('changes', [
    {'stream3_ring_slots': 4},
    {'shard_count': 2}, {'shard_count': 8},
    {'stream4_active_sort_slots': 1},
    {'stream4_batch_candidates': 65536, 'stream4_trigger_candidates': 65536},
    {'stream4_batch_candidates': 262144, 'stream4_trigger_candidates': 262144},
])
def test_downstream_trials_preserve_beam_scorer_and_full_receive_capacity(changes):
    base = runtime_plan(2**20, model_micro=64, inference_concurrency=1)
    trial = downstream_plan(base, changes)
    assert trial.effective_beam == base.effective_beam
    assert trial.runtime['model_micro'] == 64
    assert trial.runtime['stream1_concurrency'] == 1
    assert trial.parent_batch == 8192
    assert trial.shard_capacity_candidates >= (trial.stream3_batch_candidates
        + trial.runtime['stream4_batch_candidates'] + trial.runtime['stream4_trigger_candidates'])


@pytest.mark.parametrize('changes', [
    {'b_micro': 1024}, {'model_micro': 128}, {'stream1_concurrency': 2},
    {'shard_capacity_candidates': 1}, {'semantic_shard_cap': 1},
])
def test_downstream_trials_reject_scorer_or_semantic_capacity_changes(changes):
    with pytest.raises(ValueError, match='cannot change'):
        downstream_plan(runtime_plan(2**20), changes)


def test_saturated_layers_uses_parallel_max_and_ignores_initial_growth(tmp_path: Path):
    for rank in (0, 1):
        rows = ['depth_start=4 frontier_size=100\ndepth_done=4 depth_sec=999\n']
        rows.extend(f'depth_start={d} frontier_size=524288\n'
                    f'depth_done={d} depth_sec={10 if rank == 0 else d+6}\n'
                    for d in range(5, 9))
        (tmp_path/f'rank-{rank}.log').write_text(''.join(rows))
    layers = saturated_layers(tmp_path, 524288)
    assert [row['seconds'] for row in layers] == [11, 12, 13, 14]
    assert layers[0]['rank_seconds'] == [10, 11]


def test_short_paired_blocks_preserve_both_rank_times(tmp_path):
    for rank in (0, 1):
        (tmp_path/f'rank-{rank}.log').write_text(''.join(
            f'depth_start={d} frontier_size=524288\ndepth_done={d} depth_sec={10+rank}\n'
            for d in (5, 6)))
    layers = saturated_layers(tmp_path, 524288, required_layers=2)
    assert [row['rank_seconds'] for row in layers] == [[10, 11], [10, 11]]
    with pytest.raises(ValueError, match='steady-state'):
        saturated_layers(tmp_path, 524288)


def test_saturated_layers_rejects_missing_second_rank_completion(tmp_path: Path):
    for rank in (0, 1):
        rows = ''.join(f'depth_start={d} frontier_size=524288\n'
                       f'depth_done={d} depth_sec=10\n' for d in range(5, 9-rank))
        (tmp_path/f'rank-{rank}.log').write_text(rows)
    with pytest.raises(ValueError, match='steady-state'):
        saturated_layers(tmp_path, 524288)
