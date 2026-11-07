"""Saved native capture through real collector subprocess; not fresh inference."""
import json
import os
from pathlib import Path
import sys
import pytest
from tools.cube4_production_preflight import collect_fresh_candidate
from tools.cube4_numeric_gate import validate_reference

FIELDS=('stride','increment_row','increment_group','increment_cluster',
    'advance_row','advance_group','advance_cluster','advance_tile')
MODES=['valid','missing','device','execution_hash','inventory_hash','schema_bool',
    'promotion','missing_lane','duplicate_node','wrong_function','unknown_role','wrong_width']
MODES += [operand+'_'+field for operand in ('C','D') for field in FIELDS]

@pytest.mark.parametrize('mode',MODES)
def test_public_gemm_iterator_collector_rejects_corrupted_actual_receipt(mode,tmp_path):
    selected=os.environ.get('BEAM_TEST_GEMM_SCORE_OUTPUT')
    reference_dir=os.environ.get('BEAM_TEST_GEMM_REFERENCE_DIR')
    geometry_path=os.environ.get('BEAM_TEST_GEMM_ITERATOR_GEOMETRY')
    if not selected or not reference_dir or not geometry_path:
        pytest.skip('authorized remote actual captures and independent geometry required')
    source=Path(selected)
    execution=json.loads((source/'execution.json').read_bytes())
    keys=('reference_sha256','manifest_sha256','outer_microbatch','transformer_microbatch',
        'lanes','executor','device','loaded_tensor_bindings','requested_launch_policies')
    expected={key:execution[key] for key in keys}
    reference=validate_reference(json.loads((Path(reference_dir)/'reference.json').read_bytes()))
    geometry=json.loads(Path(geometry_path).read_bytes())
    def observe():
        return dict(expected=expected,reference_dir=reference_dir,reference_scores=reference,
            require_graph_inventory=True,expected_graph_gemm_iterator_geometry=geometry)
    code=f'''import pathlib,sys,shutil,json
source=pathlib.Path({str(source)!r});output=pathlib.Path(sys.argv[3]);output.mkdir()
for name in ('execution.json','raw_scores.json','graph_kernel_inventory.json'):
 shutil.copyfile(source/name,output/name)
p=json.loads((source/'graph_gemm_members.json').read_bytes());mode={mode!r}
node=next(n for n in p['lanes'][0]['kernels'] if n['gemm_signature_known'])
if mode=='device':p['device']['uuid_hex']='0'*32
elif mode=='execution_hash':p['execution_sha256']='0'*64
elif mode=='inventory_hash':p['inventory_sha256']='0'*64
elif mode=='schema_bool':p['schema_version']=True
elif mode=='promotion':p['parameter_values_checked']=True
elif mode=='missing_lane':p['lanes'].pop()
elif mode=='duplicate_node':p['lanes'][0]['kernels'][1]=p['lanes'][0]['kernels'][0]
elif mode=='wrong_function':node['function_id']+=1
elif mode=='unknown_role':node['pointers']['B']['role']='unregistered_weight'
elif mode=='wrong_width':node['problem']['n']+=1
elif mode.startswith(('C_','D_')):
 operand,field=mode.split('_',1);node['epilogue_iterators'][operand][field]+=17
if mode!='missing':(output/'graph_gemm_members.json').write_text(json.dumps(p))
'''
    if mode=='valid':
        result=collect_fresh_candidate([sys.executable,'-c',code],tmp_path/'run',10,observe,reference)
        assert result['graph_gemm_public_iterators_checked'] is True
        assert result['production_admitted'] is False
    else:
        with pytest.raises(ValueError):
            collect_fresh_candidate([sys.executable,'-c',code],tmp_path/'run',10,observe,reference)
