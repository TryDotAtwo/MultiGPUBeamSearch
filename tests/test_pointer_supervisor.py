"""Fresh subprocess sidecar transaction, real independent pointer validator."""
from copy import deepcopy
import hashlib
import json
import os
import sys
import pytest
from test_production_preflight import fixture as scores_fixture
from test_graph_pointer_contract import fixture as pointers_fixture
from tools.cube4_production_preflight import collect_fresh_candidate

@pytest.mark.skipif(os.name!='posix',reason='remote process transaction')
@pytest.mark.parametrize('mode',['valid','missing','device','execution_hash','inventory_hash','role','symbol','same_symbol_rows'])
def test_fresh_pointer_transaction(mode,tmp_path):
    ref,scores,execution,expected=scores_fixture()
    values,model,shape=pointers_fixture()
    for key in ('executor','outer_microbatch','transformer_microbatch','requested_launch_policies','lanes'):
        expected[key]=deepcopy(shape[key]);execution[key]=deepcopy(shape[key])
    expected['device']['sm']=execution['device']['sm']=75
    for job,count,base in zip(scores['jobs'],[2,1,1,0],[0,1,2,0]):
        job.update(active_rows=count,parent_base=base,reference_rows=[0]*count,
                   raw_scores=[ref[0][:] for _ in range(count)])
    execution_text=json.dumps(execution);execution_hash=hashlib.sha256(execution_text.encode()).hexdigest()
    symbols=list(dict.fromkeys(n['mangled_name'] for n in values['lanes'][0]['kernels']))
    attrs=dict(binary_version=75,ptx_version=75,max_threads_per_block=1024,num_regs=12,
        static_shared_bytes=0,local_bytes=0,constant_bytes=0,max_dynamic_shared_bytes=49152)
    rows=[114,114,114,114,114,114,114,2,2,2,114,1]
    nodes=[dict(function_id=symbols.index(n['mangled_name']),grid=[row,1,1],block=[128,1,1],dynamic_shared_bytes=0)
           for n,row in zip(values['lanes'][0]['kernels'],rows)]
    inventory=dict(schema_version=1,scope='captured_graph_kernel_inventory_not_admission',
        production_admitted=False,kernel_coverage_complete=False,device=deepcopy(expected['device']),
        execution_sha256=execution_hash,functions=[dict(mangled_name=s,attributes=attrs) for s in symbols],
        lanes=[dict(lane=0,node_count=len(nodes),node_types=dict(kernel=len(nodes),memcpy=0,memset=0),kernels=nodes)])
    inventory_text=json.dumps(inventory)
    values.update(device=deepcopy(expected['device']),execution_sha256=execution_hash,
                  inventory_sha256=hashlib.sha256(inventory_text.encode()).hexdigest())
    if mode=='device':values['device']['uuid_hex']='0'*32
    if mode=='execution_hash':values['execution_sha256']='0'*64
    if mode=='inventory_hash':values['inventory_sha256']='0'*64
    if mode=='role':values['lanes'][0]['kernels'][0]['pointers'][0]['role']='lane_logits'
    if mode=='symbol':
        # Swap valid whole nodes but preserve positions: profile multiset remains
        # valid; only binding to actual native inventory must reject this.
        nodes=values['lanes'][0]['kernels'];nodes[0],nodes[1]=nodes[1],nodes[0]
        nodes[0]['kernel_index']=0;nodes[1]['kernel_index']=1
    if mode=='same_symbol_rows':
        nodes=values['lanes'][0]['kernels'];nodes[1],nodes[8]=nodes[8],nodes[1]
        nodes[1]['kernel_index']=1;nodes[8]['kernel_index']=8
    files={'execution.json':execution_text,'raw_scores.json':json.dumps(scores),
           'graph_kernel_inventory.json':inventory_text}
    if mode!='missing':files['graph_pointer_roles.json']=json.dumps(values)
    code='import pathlib,sys\nout=pathlib.Path(sys.argv[3]);out.mkdir()\n'
    for name,text in files.items():code+=f'(out/{name!r}).write_text({text!r})\n'
    def observe():
        return dict(expected=deepcopy(expected),reference_dir=str(tmp_path),reference_scores=deepcopy(ref),
            require_graph_inventory=True,expected_graph_pointer_profile=dict(model=model,padded_seq_len=57,ln_shared_bytes=16))
    if mode=='valid':
        result=collect_fresh_candidate([sys.executable,'-c',code],tmp_path/'run',5,observe,ref)
        assert result['graph_known_pointer_roles_checked'] is True
        assert result['production_admitted'] is False
    else:
        with pytest.raises(ValueError):
            collect_fresh_candidate([sys.executable,'-c',code],tmp_path/'run',5,observe,ref)
