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


@pytest.mark.parametrize('beam', [8, 65536, 1048576])
def test_profile_reserves_incoming_batch_and_retains_requested_beam(beam):
    p = runtime_plan(beam)
    assert p.effective_beam == beam
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
