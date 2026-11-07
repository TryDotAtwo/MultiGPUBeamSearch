"""Actual captured FMHA Params evidence, not host launch claims or admission."""
import hashlib
import json
import os
from pathlib import Path
import pytest
from collections import Counter
from tools.attention_member_values import validate_attention_params


def test_actual_scorer_publishes_captured_attention_params():
    selected = os.environ.get('BEAM_TEST_ATTENTION_SCORE_OUTPUT')
    if not selected:
        pytest.skip('explicit authorized remote actual scorer output required')
    root = Path(selected)
    execution_raw = (root/'execution.json').read_bytes()
    inventory_raw = (root/'graph_kernel_inventory.json').read_bytes()
    execution = json.loads(execution_raw)
    inventory = json.loads(inventory_raw)
    path = root/'graph_attention_members.json'
    assert path.is_file(), 'real scorer did not publish captured FMHA Params members'
    payload = json.loads(path.read_bytes())
    assert payload['schema_version'] == 1
    assert payload['scope'] == 'captured_attention_params_not_gemm_or_admission'
    assert payload['production_admitted'] is False
    assert payload['parameter_values_checked'] is False
    assert payload['device'] == execution['device']
    assert payload['execution_sha256'] == hashlib.sha256(execution_raw).hexdigest()
    assert payload['inventory_sha256'] == hashlib.sha256(inventory_raw).hexdigest()
    assert len(payload['lanes']) == execution['lanes']
    # Independent diagnostic scope: member presence is a producer gate, not a
    # replacement for independently validating values, extents and geometry.
    captures = {lane['lane']: lane['kernels'] for lane in inventory['lanes']}
    seen = set()
    padded = int(os.environ['BEAM_TEST_ATTENTION_PADDED_SEQ'])
    policies = execution['requested_launch_policies']
    final_cls = (policies.get('BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY') == '1' and
                 policies.get('BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION') == '1')
    split = policies.get('BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV') == '1'
    expected_counts = Counter()
    for offset in range(0, execution['outer_microbatch'], execution['transformer_microbatch']):
        batch = min(execution['transformer_microbatch'], execution['outer_microbatch']-offset)
        expected_counts[(batch, False)] += 3 if final_cls else 4
        if final_cls: expected_counts[(batch, True)] += 1
    for lane in payload['lanes']:
        lid = lane['lane']
        assert lid in captures and lid not in seen
        seen.add(lid)
        assert len(lane['kernels']) == len(captures[lid])
        known = 0
        counts = Counter()
        for index, node in enumerate(lane['kernels']):
            assert node['kernel_index'] == index
            assert node['function_id'] == captures[lid][index]['function_id']
            symbol = inventory['functions'][node['function_id']]['mangled_name']
            attention = symbol.startswith('_Z29attention_kernel_batched_impl')
            assert node['attention_signature_known'] is attention
            if attention:
                known += 1
                assert node['parameter_index'] == 0
                assert set(node['pointers']) == {
                    'query_ptr', 'key_ptr', 'value_ptr', 'output_ptr',
                    'output_accum_ptr', 'logsumexp_ptr'}
                assert set(node['scalars']) >= {
                    'scale','num_heads','num_batches','head_dim','head_dim_value',
                    'num_queries','num_keys','custom_mask_type','q_strideH',
                    'k_strideH','v_strideH','q_strideM','k_strideM','v_strideM',
                    'q_strideB','k_strideB','v_strideB','o_strideM'}
                batch = captures[lid][index]['grid'][2]
                cls = node['scalars']['num_queries'] == 1
                assert validate_attention_params(node['pointers'],node['scalars'],
                    batch=batch,seq=padded,cls=cls,split=split and cls)
                counts[(batch,cls)] += 1
            else:
                assert node['parameter_index'] is None
                assert node['pointers'] == {} and node['scalars'] == {}
        assert known > 0
        assert counts == expected_counts
