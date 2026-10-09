"""Admission must reject bad consumed NetworkView members, not only Dims.

The break caught here is ignoring the independently required embedded-pointer
profile, including equal-sized gamma/beta swaps that extent-only checks miss.
These are subprocess protocol tests, not evidence of native CUDA extraction.
"""
from copy import deepcopy
import hashlib
import json
import os
import sys
import pytest
from test_production_preflight import fixture as scores_fixture
from test_graph_scalar_contract import fixture as graph_fixture
from tools.cube4_production_preflight import collect_fresh_candidate


@pytest.mark.skipif(os.name != 'posix', reason='remote process transaction')
@pytest.mark.parametrize('mode', [
    'valid', 'missing', 'null', 'gamma_beta_swap', 'short_extent',
    'wrong_index', 'device', 'execution_hash', 'inventory_hash',
    'missing_lane', 'duplicate_lane', 'missing_node', 'duplicate_node',
    'wrong_function', 'opaque_promoted', 'missing_member', 'duplicate_member',
    'member_offset', 'bool_extent', 'bool_kernel_index', 'bool_network_index',
    'bool_schema', 'scope', 'admission', 'parameter_promotion', 'attention_missing',
])
def test_fresh_consumed_network_members(mode, tmp_path):
    ref, scores, execution, expected = scores_fixture()
    _, inventory, model, shape = graph_fixture()
    for key in ('executor', 'outer_microbatch', 'transformer_microbatch',
                'requested_launch_policies', 'lanes'):
        expected[key] = deepcopy(shape[key])
        execution[key] = deepcopy(shape[key])
    expected['device']['sm'] = execution['device']['sm'] = 75
    for job, count, base in zip(scores['jobs'], [32, 31, 1, 0], [0, 1, 32, 0]):
        job.update(active_rows=count, parent_base=base,
                   reference_rows=[0] * count,
                   raw_scores=[ref[0][:] for _ in range(count)])
    execution_text = json.dumps(execution)
    execution_hash = hashlib.sha256(execution_text.encode()).hexdigest()
    inventory.update(schema_version=1,
        scope='captured_graph_kernel_inventory_not_admission',
        production_admitted=False, kernel_coverage_complete=False,
        device=deepcopy(expected['device']), execution_sha256=execution_hash)
    inventory_text = json.dumps(inventory)
    # Literal FP16 Cube4 extents, independently checked against loader formulas.
    members = [dict(member=name, role=name, offset=0, available_bytes=size)
        for name, size in [('fast_slot_projected', 9216),
            ('fast_piece_static', 28672), ('cls_token', 512),
            ('input_ln_gamma', 512), ('input_ln_beta', 512),
            ('piece_positions', 336), ('piece_mask', 168)]]
    nodes = []
    for index, node in enumerate(inventory['lanes'][0]['kernels']):
        fid = node['function_id']
        known = fid == 0  # Literal fixture mapping: fused input function.
        nodes.append(dict(kernel_index=index, function_id=fid,
            network_signature_known=known, network_index=4 if known else None,
            members=deepcopy(members) if known else []))
    payload = dict(schema_version=1,
        scope='captured_consumed_network_members_not_opaque_params_or_admission',
        production_admitted=False, parameter_values_checked=False,
        device=deepcopy(expected['device']), execution_sha256=execution_hash,
        inventory_sha256=hashlib.sha256(inventory_text.encode()).hexdigest(),
        lanes=[dict(lane=0, kernels=nodes)])
    if mode == 'null': nodes[0]['members'][0].update(role='null', available_bytes=0)
    if mode == 'gamma_beta_swap':
        nodes[0]['members'][3]['role'] = 'input_ln_beta'
        nodes[0]['members'][4]['role'] = 'input_ln_gamma'
    if mode == 'short_extent': nodes[0]['members'][0]['available_bytes'] = 9215
    if mode == 'wrong_index': nodes[0]['network_index'] = 5
    if mode == 'device': payload['device']['uuid_hex'] = '0' * 32
    if mode == 'execution_hash': payload['execution_sha256'] = '0' * 64
    if mode == 'inventory_hash': payload['inventory_sha256'] = '0' * 64
    # These catch omitted coverage and type checks, not source-text changes.
    if mode == 'missing_lane': payload['lanes'] = []
    if mode == 'duplicate_lane': payload['lanes'].append(deepcopy(payload['lanes'][0]))
    if mode == 'missing_node': nodes.pop()
    if mode == 'duplicate_node': nodes[-1] = deepcopy(nodes[0])
    if mode == 'wrong_function': nodes[0]['function_id'] = 1
    if mode == 'opaque_promoted':
        nodes[1].update(network_signature_known=True, network_index=4,
                        members=deepcopy(members))
    if mode == 'missing_member': nodes[0]['members'].pop()
    if mode == 'duplicate_member': nodes[0]['members'][-1] = deepcopy(nodes[0]['members'][0])
    if mode == 'member_offset': nodes[0]['members'][0]['offset'] = 1
    if mode == 'bool_extent': nodes[0]['members'][0]['available_bytes'] = True
    if mode == 'bool_kernel_index': nodes[0]['kernel_index'] = False
    if mode == 'bool_network_index': nodes[0]['network_index'] = True
    if mode == 'bool_schema': payload['schema_version'] = True
    if mode == 'scope': payload['scope'] = 'production'
    if mode == 'admission': payload['production_admitted'] = True
    if mode == 'parameter_promotion': payload['parameter_values_checked'] = True
    files = {'execution.json': execution_text, 'raw_scores.json': json.dumps(scores),
             'graph_kernel_inventory.json': inventory_text}
    if mode != 'missing': files['graph_network_members.json'] = json.dumps(payload)
    code = 'import pathlib,sys\nout=pathlib.Path(sys.argv[3]);out.mkdir()\n'
    for name, value in files.items():
        code += f'(out/{name!r}).write_text({value!r})\n'
    def observe():
        observed = dict(expected=deepcopy(expected), reference_dir=str(tmp_path),
            reference_scores=deepcopy(ref), require_graph_inventory=True,
            expected_graph_network_profile=dict(model=model,
                padded_seq_len=57, ln_shared_bytes=16))
        if mode == 'attention_missing':
            observed['expected_graph_attention_members_profile'] = dict(model=model,padded_seq_len=57)
        return observed
    if mode == 'valid':
        receipt = collect_fresh_candidate([sys.executable, '-c', code],
            tmp_path / 'run', 5, observe, ref)
        assert receipt.get('graph_consumed_network_members_checked') is True
        assert receipt['production_admitted'] is False
    else:
        with pytest.raises(ValueError):
            collect_fresh_candidate([sys.executable, '-c', code],
                tmp_path / 'run', 5, observe, ref)
