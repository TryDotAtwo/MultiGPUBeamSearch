from dataclasses import replace
from pathlib import Path

from tools.cube555.run import runtime_plan
from tools.cayleypy_public.config import PublicRunConfig
from tools.cayleypy_public.runner import build_runner_invocation, maximum_history_depth


def test_one_real_rank_owns_entire_aligned_beam():
    one = runtime_plan(3_100_000, world_size=1)
    two = runtime_plan(3_100_000)
    assert one.world_size == 1 and one.local_beam == one.effective_beam >= 3_100_000
    assert two.world_size == 2 and two.local_beam * 2 == two.effective_beam


def test_single_rank_history_uses_whole_budget():
    plan = runtime_plan(1_048_576, world_size=1)
    single = maximum_history_depth(plan, 30, 5, 8_000_000_000, 30_000_000_000)
    half = maximum_history_depth(replace(plan, world_size=2), 30, 5,
                                 8_000_000_000, 30_000_000_000)
    assert single > half


def test_single_rank_torchrun_and_log_contract(tmp_path, monkeypatch):
    from tools.cayleypy_public import runner
    monkeypatch.setattr(runner, 'preflight_history_runtime', lambda *a, **kw: None)
    config = PublicRunConfig.from_mapping(dict(author_name='test', checkpoint_path=tmp_path/'m.pt',
        puzzle_info_json=tmp_path/'p.json', test_csv=tmp_path/'t.csv',
        sample_submission_csv=tmp_path/'s.csv', puzzle_id_start=0, puzzle_id_end=0,
        beam_width=65536, max_depth=6, reflect_mode='off', reflect_source_csv=None,
        solution_mode='first', collect_until_depth=0, max_collected_solutions=2000,
        touch_bfs_radius=0, publish_results=False, results_ingest_url='https://example.com'))
    invocation = build_runner_invocation(config, runtime_plan(65536, world_size=1), 30, 0,
                                 'original', tmp_path, tmp_path/'logs')
    assert '--nproc-per-node=1' in invocation.command
    assert len(invocation.rank_logs) == 1 and invocation.rank_logs[0].name == 'rank-0.log'
