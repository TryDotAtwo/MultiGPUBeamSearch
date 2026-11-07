"""Fresh native capture through actual collector; mutation replay is not inference."""
import copy,json,os,sys
from pathlib import Path
import pytest
from tools.cube4_production_preflight import collect_fresh_candidate
from tools.cube4_numeric_gate import validate_reference

FIELDS=('stride','increment_strided','increment_next','advance')
MODES=['valid','missing','device','execution_hash','inventory_hash','wrong_k','wrong_role',
    'schema_bool','extra_field','bool_field','missing_geometry']
MODES += [operand+'_'+field for operand in ('A','B') for field in FIELDS]

@pytest.mark.parametrize('mode',MODES)
def test_mainloop_collector_rejects_corrupted_actual_receipt(mode,tmp_path):
    selected=os.environ.get('BEAM_TEST_GEMM_SCORE_OUTPUT')
    reference_dir=os.environ.get('BEAM_TEST_GEMM_REFERENCE_DIR')
    geometry_path=os.environ.get('BEAM_TEST_GEMM_ITERATOR_GEOMETRY')
    if not selected or not reference_dir or not geometry_path:
        pytest.skip('authorized fresh native capture and independent geometry required')
    source=Path(selected);execution=json.loads((source/'execution.json').read_bytes())
    keys=('reference_sha256','manifest_sha256','outer_microbatch','transformer_microbatch',
        'lanes','executor','device','loaded_tensor_bindings','requested_launch_policies')
    expected={key:execution[key] for key in keys}
    reference=validate_reference(json.loads((Path(reference_dir)/'reference.json').read_bytes()))
    geometry=json.loads(Path(geometry_path).read_bytes())
    if mode=='missing_geometry':
        geometry=copy.deepcopy(geometry);geometry['regular'].pop('mainloop')
    def observe():
        return dict(expected=expected,reference_dir=reference_dir,reference_scores=reference,
            require_graph_inventory=True,expected_graph_gemm_mainloop_geometry=geometry)
    code=f'''import pathlib,sys,shutil,json
source=pathlib.Path({str(source)!r});output=pathlib.Path(sys.argv[3]);output.mkdir()
for name in ('execution.json','raw_scores.json','graph_kernel_inventory.json'):
    shutil.copyfile(source/name,output/name)
p=json.loads((source/'graph_gemm_members.json').read_bytes())
nodes=[n for l in p['lanes'] for n in l['kernels'] if n['gemm_signature_known']]
n=nodes[0];mode={mode!r}
if mode=='missing':n.pop('mainloop_iterators',None)
elif mode=='device':p['device']['uuid']='wrong-device'
elif mode=='execution_hash':p['execution_sha256']='0'*64
elif mode=='inventory_hash':p['inventory_sha256']='0'*64
elif mode=='wrong_k':n['problem']['k']+=17
elif mode=='wrong_role':n['pointers']['B']['role']='unknown-weight'
elif mode=='schema_bool':p['schema_version']=True
elif mode=='extra_field':n['mainloop_iterators']['A']['other']=0
elif mode=='bool_field':n['mainloop_iterators']['A']['stride']=True
elif mode not in ('valid','missing_geometry'):
    operand,field=mode.split('_',1);n['mainloop_iterators'][operand][field]+=17
(output/'graph_gemm_members.json').write_text(json.dumps(p))
'''
    command=[sys.executable,'-c',code]
    if mode=='valid':
        result=collect_fresh_candidate(command,tmp_path/'run',30,observe,reference)
        assert result.get('graph_gemm_mainloop_iterators_checked') is True
        assert result['production_admitted'] is False
    else:
        with pytest.raises(ValueError):collect_fresh_candidate(command,tmp_path/'run',30,observe,reference)
