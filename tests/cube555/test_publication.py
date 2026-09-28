import copy
import json
import pytest
import runpy
from pathlib import Path
_helpers = runpy.run_path(str(Path(__file__).parents[1] / 'cayleypy_public/test_results.py'))
_context, _solution = _helpers['_context'], _helpers['_solution']
from tools.cayleypy_public.results import build_result_envelope

def inputs():
    context, solution = _context(), _solution()
    goal = list(range(150))
    permutation = goal.copy()
    permutation[0], permutation[149] = 149, 0
    context['proof'] = dict(initial_state=permutation, central_state=goal, generators={'swap': permutation})
    context['model'] = dict(filename='cube555_blend.ts', sha256='a'*64, format='cube555-q-blend', manifest=dict(state_len=150, num_classes=150, output_dim=1, dtype='fp16', model_family='cube555_q_blend', blend_weights=[0.8, 0.2], checkpoint_sha256=['b'*64,'c'*64], script_sha256='a'*64, score_contract='parent_q_fp32_blend_then_clamp_round'))
    context['profile']['model_class'] = 'output1'
    solution.update(path='swap', original_oriented_path='swap', reached_state=goal, found_depth=1, touch_depth=0, variant='original', valid=True)
    return context, solution

def test_high_index_150_facelet_replay_and_blend_provenance():
    c,s=inputs()
    envelope=build_result_envelope(c,s)
    assert envelope['proof']['initial_state'][0] == 149
    assert len(envelope['model']['manifest']['checkpoint_sha256']) == 2
    assert envelope['model']['manifest']['blend_weights'] == [0.8,0.2]

def test_false_555_solution_rejected():
    c,s=inputs(); s['path']=''; s['original_oriented_path']=''
    with pytest.raises(ValueError):build_result_envelope(c,s)
