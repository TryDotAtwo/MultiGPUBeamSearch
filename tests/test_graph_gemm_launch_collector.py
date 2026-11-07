"""Real collector with captured native receipt mutations; no fresh inference claim."""
import json
import os
from pathlib import Path
import sys
import pytest
from tools.cube4_production_preflight import collect_fresh_candidate
from tools.cube4_numeric_gate import validate_reference


@pytest.mark.parametrize('mode', ['valid', 'missing', 'grid', 'batch', 'k', 'row',
                                  'duplicate', 'scope', 'hash', 'unknown',
                                  'alpha', 'stride', 'pointer', 'head_beta', 'residual_beta',
                                  'array_alpha_missing', 'array_beta_missing',
                                  'array_alpha_nonnull', 'array_beta_nonnull',
                                  'array_alpha_bool', 'array_beta_bool',
                                  'override', 'override_bool', 'tensor_missing', 'tensor_bool',
                                  'tensor_stride', 'tensor_increment_row', 'tensor_increment_group',
                                  'tensor_increment_cluster', 'tensor_advance_row',
                                  'tensor_advance_group', 'tensor_advance_cluster', 'tensor_advance_tile'])
def test_launch_gate_in_actual_collector(mode, tmp_path):
    capture = os.environ.get('BEAM_TEST_GEMM_SCORE_OUTPUT')
    reference_dir = os.environ.get('BEAM_TEST_GEMM_REFERENCE_DIR')
    geometry_path = os.environ.get('BEAM_TEST_GEMM_ITERATOR_GEOMETRY')
    if not all((capture, reference_dir, geometry_path)):
        pytest.skip('explicit remote native capture and compiled geometry required')
    source = Path(capture)
    execution = json.loads((source / 'execution.json').read_bytes())
    keys = ('reference_sha256', 'manifest_sha256', 'outer_microbatch',
            'transformer_microbatch', 'lanes', 'executor', 'device',
            'loaded_tensor_bindings', 'requested_launch_policies')
    expected = {key: execution[key] for key in keys}
    reference = validate_reference(json.loads((Path(reference_dir) / 'reference.json').read_bytes()))
    geometry = json.loads(Path(geometry_path).read_bytes())
    def observe():
        return dict(expected=expected, reference_dir=reference_dir,
                    reference_scores=reference, require_graph_inventory=True,
                    expected_graph_gemm_launch_profile=dict(geometry=geometry, padded_seq_len=64))
    code = f'''import pathlib,sys,shutil,json
source=pathlib.Path({str(source)!r});output=pathlib.Path(sys.argv[3]);output.mkdir()
for name in ('execution.json','raw_scores.json','graph_kernel_inventory.json'):
 shutil.copyfile(source/name,output/name)
p=json.loads((source/'graph_gemm_members.json').read_bytes());mode={mode!r}
node=next(n for n in p['lanes'][0]['kernels'] if n['gemm_signature_known'] and 'batch_count' in n['launch_scalars'])
if mode=='missing':node.pop('launch_scalars')
elif mode=='grid':node['launch_scalars']['grid_tiled_shape']['m']+=1
elif mode=='batch':node['launch_scalars']['batch_count']=2
elif mode=='k':node['launch_scalars']['gemm_k_size']=1024
elif mode=='row':node['problem']['m']+=128
elif mode=='duplicate':p['lanes'][0]['kernels'][1]=p['lanes'][0]['kernels'][0]
elif mode=='scope':p['parameter_values_checked']=True
elif mode=='hash':p['inventory_sha256']='0'*64
elif mode=='unknown':node['launch_scalars']['extra']=0
elif mode=='alpha':node['scalars']['alpha']=0.0
elif mode=='stride':node['scalars']['batch_stride_A']+=256
elif mode=='pointer':node['pointers']['D']['offset']=2
elif mode=='override':node['epilogue_pointers']['alpha']['role']='lane_context'
elif mode=='override_bool':node['epilogue_pointers']['beta']['offset']=False
elif mode.startswith('array_'):
 regular=next(n for n in p['lanes'][0]['kernels'] if n['gemm_signature_known'] and 'batch_count' not in n['launch_scalars'])
 _,coefficient,mutation=mode.split('_');field=coefficient+'_array'
 if mutation=='missing':regular['epilogue_pointers'].pop(field)
 elif mutation=='nonnull':regular['epilogue_pointers'][field]['role']='lane_context'
 else:regular['epilogue_pointers'][field]['offset']=False
elif mode=='tensor_missing':node.pop('tensor_iterator')
elif mode=='tensor_bool':node['tensor_iterator']['stride']=False
elif mode.startswith('tensor_'):node['tensor_iterator'][mode[7:]]+=2
elif mode=='head_beta':
 head=next(n for n in p['lanes'][0]['kernels'] if n['gemm_signature_known'] and n['pointers']['B']['role']=='output_weight')
 head['scalars']['beta']=1.0
elif mode=='residual_beta':
 residual=next(n for n in p['lanes'][0]['kernels'] if n['gemm_signature_known'] and n['pointers']['B']['role'].endswith('_ff2_weight'))
 residual['scalars']['beta']=0.0
(output/'graph_gemm_members.json').write_text(json.dumps(p))
'''
    if mode == 'valid':
        result = collect_fresh_candidate([sys.executable, '-c', code], tmp_path / 'run', 15, observe, reference)
        assert result['graph_gemm_launch_values_checked'] is True
        assert result['graph_gemm_tensor_iterators_checked'] is True
        assert result['production_admitted'] is False
        assert result.get('parameter_values_checked', False) is False
    else:
        with pytest.raises(ValueError):
            collect_fresh_candidate([sys.executable, '-c', code], tmp_path / 'run', 15, observe, reference)
