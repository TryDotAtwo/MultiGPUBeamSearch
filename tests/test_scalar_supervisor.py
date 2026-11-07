"""Real subprocess transaction, real scalar validator, hand-derived values."""
from copy import deepcopy
import hashlib
import json
import os
import sys
import pytest
from test_production_preflight import fixture as scores_fixture
from test_graph_scalar_contract import fixture as scalars_fixture
from tools.cube4_production_preflight import collect_fresh_candidate


@pytest.mark.skipif(os.name!='posix',reason='remote process transaction')
@pytest.mark.parametrize('mode',['valid','missing','device','execution_hash','inventory_hash','value'])
def test_fresh_scalar_sidecar_transaction(mode,tmp_path):
    ref,scores,execution,expected=scores_fixture()
    values,inventory,model,shape=scalars_fixture()
    for key in ('executor','outer_microbatch','transformer_microbatch','requested_launch_policies','lanes'):
        expected[key]=deepcopy(shape[key]);execution[key]=deepcopy(shape[key])
    expected['device']['sm']=execution['device']['sm']=75
    for job,count,base in zip(scores['jobs'],[32,31,1,0],[0,1,32,0]):
        job.update(active_rows=count,parent_base=base,reference_rows=[0]*count,
                   raw_scores=[ref[0][:] for _ in range(count)])
    execution_text=json.dumps(execution);execution_hash=hashlib.sha256(execution_text.encode()).hexdigest()
    inventory.update(schema_version=1,scope='captured_graph_kernel_inventory_not_admission',
        production_admitted=False,kernel_coverage_complete=False,
        device=deepcopy(expected['device']),execution_sha256=execution_hash)
    inventory_text=json.dumps(inventory)
    values.update(device=deepcopy(expected['device']),execution_sha256=execution_hash,
                  inventory_sha256=hashlib.sha256(inventory_text.encode()).hexdigest())
    if mode=='device':values['device']['uuid_hex']='0'*32
    if mode=='execution_hash':values['execution_sha256']='0'*64
    if mode=='inventory_hash':values['inventory_sha256']='0'*64
    if mode=='value':values['lanes'][0]['kernels'][-1]['u32_values'][-1]['value']=0
    files={'execution.json':execution_text,'raw_scores.json':json.dumps(scores),
           'graph_kernel_inventory.json':inventory_text}
    if mode!='missing':files['graph_u32_values.json']=json.dumps(values)
    code='import pathlib,sys\nout=pathlib.Path(sys.argv[3]);out.mkdir()\n'
    for name,text in files.items():code+=f'(out/{name!r}).write_text({text!r})\n'
    def observe():
        return dict(expected=deepcopy(expected),reference_dir=str(tmp_path),
            reference_scores=deepcopy(ref),require_graph_inventory=True,
            expected_graph_non_gemm_scalar_profile=dict(model=model,padded_seq_len=57,ln_shared_bytes=16))
    if mode=='valid':
        result=collect_fresh_candidate([sys.executable,'-c',code],tmp_path/'run',5,observe,ref)
        assert result['graph_non_gemm_u32_values_checked'] is True
        assert result['production_admitted'] is False
    else:
        with pytest.raises(ValueError):
            collect_fresh_candidate([sys.executable,'-c',code],tmp_path/'run',5,observe,ref)
