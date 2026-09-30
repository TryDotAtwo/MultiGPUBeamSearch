import json
from pathlib import Path
import pandas as pd
import pytest
from tools.cube555.models import Blend
from tools.cube555.run import runtime_plan
from tools.cayleypy_public.data import load_puzzle_contract


def test_150_facelets_preserve_high_values(tmp_path):
    goal = list(range(150))
    perm = goal[1:] + goal[:1]
    (tmp_path / 'info.json').write_text(json.dumps(dict(central_state=goal, generators={'r': perm})))
    pd.DataFrame([dict(initial_state_id=7, initial_state=','.join(map(str, perm)))]).to_csv(tmp_path / 'test.csv', index=False)
    pd.DataFrame([dict(initial_state_id=7, solution='r')]).to_csv(tmp_path / 'sample.csv', index=False)
    result = load_puzzle_contract(tmp_path / 'info.json', tmp_path / 'test.csv', tmp_path / 'sample.csv', 7, 7)
    assert result.initial_states[7][-2] == 149
    assert result.num_classes == result.state_len == 150


@pytest.mark.parametrize('beam', [65536, 1048576, 4_000_000])
def test_profile_reserves_incoming_batch_and_retains_requested_beam(beam):
    p = runtime_plan(beam)
    assert p.requested_beam == beam
    assert beam <= p.effective_beam < beam + 16384
    assert p.stream3_batch_candidates == p.parent_batch * 30 * p.runtime['stream3_ring_slots']
    assert p.shard_capacity_candidates >= p.stream3_batch_candidates + p.runtime['stream4_batch_candidates'] + p.runtime['stream4_trigger_candidates']


@pytest.mark.parametrize('weight', [float('nan'), float('inf'), -0.1, 1.1])
def test_bad_blend_weight_rejected(weight):
    with pytest.raises(ValueError):
        Blend(None, None, weight)


def test_launcher_passes_complete_public_config(tmp_path):
    from types import SimpleNamespace
    from tools.cube555.run import configuration
    args = SimpleNamespace(assets=tmp_path, competition=tmp_path, beam=65536, depth=200, touch_radius=2)
    config = configuration(args, 1020, tmp_path / 'puzzle_info.json')
    assert config.puzzle_ids == (1020,)
    assert config.reflect_source_csv is None
    assert not config.publish_results


def test_outer_batch_and_inference_microbatch_are_independent():
    from tools.cayleypy_public.runner import _runtime_env
    from tools.cayleypy_public.model import ExportedModel
    from types import SimpleNamespace
    beam = 4_000_000
    a = runtime_plan(beam, b_micro=8192, model_micro=128)
    b = runtime_plan(beam, b_micro=8192, model_micro=512)
    assert a.effective_beam == b.effective_beam
    assert a.requested_beam == b.requested_beam == beam
    assert a.parent_batch == b.parent_batch == 8192
    assert a.shard_capacity_candidates == b.shard_capacity_candidates
    assert a.runtime['shard_count'] == 4
    assert a.runtime['stream4_batch_candidates'] > 0
    config = SimpleNamespace(puzzle_info_json='info.json', test_csv='test.csv', touch_bfs_radius=2, depth_log_every=1, puzzle_log_every=1)
    model = ExportedModel('cube555-q-blend','fp16','test',{'source_generators':'info.json'},'piece_transformer')
    env = _runtime_env(config,b,Path('weights'),model)
    assert env['BEAM_B_MICRO']=='8192'
    assert env['BEAM_STREAM1_TRANSFORMER_MICRO']=='512'


@pytest.mark.parametrize('micro',[0,-1,8193,True])
def test_bad_model_microbatch_rejected(micro):
    with pytest.raises(ValueError):
        runtime_plan(4_000_000, model_micro=micro)


def test_custom_checkpoint_is_used_in_public_config(tmp_path):
    from types import SimpleNamespace
    from tools.cube555.run import configuration
    checkpoint = tmp_path / 'replacement.pt'
    args = SimpleNamespace(assets=tmp_path, competition=tmp_path, checkpoint=checkpoint,
                           beam=2**24, depth=100, touch_radius=2)
    assert configuration(args, 1020, tmp_path / 'puzzle_info.json').checkpoint_path == checkpoint


def test_notebook_first_cell_is_simple_user_config(tmp_path):
    from tools.cube555.notebook import build
    build('a' * 40, tmp_path, False)
    notebook = json.loads((tmp_path / 'cube555-2xt4-blend.ipynb').read_text())
    first = notebook['cells'][0]
    assert first['cell_type'] == 'code'
    config = {}
    exec(''.join(first['source']), config)
    assert config['BEAM_WIDTH'] == 4_000_000
    assert config['SOLUTION_MODE'] == 'collect'
    assert 'BEAM_PROFILE' not in config
    assert 'TOUCH_BFS_RADIUS' not in config
    assert config['CHECKPOINT_PATH'].parent == config['MODEL_ROOT']
    source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert '"--checkpoint", str(CHECKPOINT_PATH)' in source
    assert '"--mlp-checkpoint", str(RESMLP_CHECKPOINT_PATH)' in source


@pytest.mark.parametrize('mode', ['off', 'after_original', 'only'])
def test_reflection_and_collection_config_reaches_runner(tmp_path, mode):
    from types import SimpleNamespace
    from tools.cube555.run import configuration
    args = SimpleNamespace(assets=tmp_path, competition=tmp_path, beam=2**22,
        depth=80, touch_radius=4, reflect_mode=mode,
        reflect_source_csv=tmp_path / 'solutions.csv', solution_mode='collect',
        collect_until_depth=100, max_collected_solutions=100)
    cfg = configuration(args, 1020, tmp_path / 'puzzle_info.json')
    assert cfg.reflect_mode == mode
    assert cfg.reflect_source_csv == args.reflect_source_csv
    assert cfg.solution_mode == 'collect'
    assert cfg.collect_until_depth == 80
    assert cfg.max_collected_solutions == 100


def test_cube555_history_reserves_runtime_memory():
    from tools.cube555.run import history_budgets
    from tools.cayleypy_public.runner import maximum_history_depth
    ram, disk = history_budgets(31_000_000_000, 80 * 1024**3)
    assert ram == 24_000_000_000
    assert maximum_history_depth(runtime_plan(4_000_000), 30, 5, ram, disk) >= 140
    ram, disk = history_budgets(20_000_000_000, 80 * 1024**3)
    assert ram == 14_000_000_000
    assert maximum_history_depth(runtime_plan(4_000_000), 30, 5, ram, disk) >= 140


def test_telemetry_preserves_collect_drop_counts(tmp_path):
    from tools.cube555.telemetry import Telemetry
    (tmp_path / 'rank-0.log').write_text('collection_truncated=1 depth=8 rank=0 hits=120 stored=100 dropped=20\n')
    (tmp_path / 'rank-1.log').write_text('collection_truncated=1 depth=8 rank=1 hits=110 stored=100 dropped=10\n')
    monitor = Telemetry(tmp_path)
    monitor.thread = type('Stopped', (), {'join': lambda self, timeout: None})()
    result = monitor.finish()
    assert result['collection_truncated'] is True
    assert result['collection_dropped_hits'] == 30
    assert len(result['collection_truncations']) == 2

@pytest.mark.parametrize('beam', [4_000_001, 2**22, 2**26, 0, True])
def test_cube555_rejects_beam_outside_supported_range(beam):
    with pytest.raises(ValueError, match='Cube555 beam'):
        runtime_plan(beam)


@pytest.mark.parametrize('identity, expected_name', [
    ({'kaggle_owner': 'anotheruser', 'kaggle_username': 'anotheruser'}, 'anotheruser'),
    ({'kaggle_owner': 'anotheruser'}, 'anotheruser'),
    ({'kaggle_owner': 'anotheruser', 'author_name': 'Display Name'}, 'Display Name'),
])
def test_fork_publication_author_is_not_hardcoded_to_original_owner(tmp_path, identity, expected_name):
    from types import SimpleNamespace
    from tools.cube555.run import configuration
    from tools.run_cayleypy_public import _publication_context
    publication = dict(competition='cayley-py-555-cube', kaggle_slug='forked-cube555',
        kaggle_version=1, solver_commit='b' * 40, kaggle_notebook_sha256='c' * 64, **identity)
    original = publication.copy()
    args = SimpleNamespace(assets=tmp_path, competition=tmp_path, beam=65536,
        depth=140, touch_radius=5, publication=publication)
    cfg = configuration(args, 35, tmp_path / 'puzzle_info.json')
    contract = SimpleNamespace(central_state=tuple(range(150)), generators={'r': tuple(range(150))}, state_len=150, move_count=1)
    model = SimpleNamespace(format='cube555-q-blend', checkpoint_sha256='a' * 64, manifest={})
    payload = _publication_context(cfg, contract, model, {}, runtime_plan(65536),
        ['Tesla T4', 'Tesla T4'], wall_seconds=1, solve_seconds=1)
    assert payload['author']['name'] == expected_name
    assert payload['kaggle']['owner'] == 'anotheruser'
    assert payload['author'].get('kaggle_username') == identity.get('kaggle_username')
    assert publication == original  # Repeated configuration must preserve the caller's metadata.
