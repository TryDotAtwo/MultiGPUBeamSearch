"""Real fresh subprocess transaction for captured embedded dimensions."""
from copy import deepcopy
import hashlib
import json
import os
import sys
import pytest
from test_production_preflight import fixture as scores_fixture
from test_graph_scalar_contract import fixture as graph_fixture
from tools.cube4_production_preflight import collect_fresh_candidate

@pytest.mark.skipif(os.name!='posix',reason='remote process transaction')
@pytest.mark.parametrize('mode',['valid','missing','dtype','state_len','bool_value','device','inventory_hash'])
def test_fresh_embedded_dims_transaction(mode,tmp_path):
    ref,scores,execution,expected=scores_fixture()
    _,inventory,model,shape=graph_fixture()
    for key in ('executor','outer_microbatch','transformer_microbatch','requested_launch_policies','lanes'):
        expected[key]=deepcopy(shape[key]);execution[key]=deepcopy(shape[key])
    expected['device']['sm']=execution['device']['sm']=75
    for job,count,base in zip(scores['jobs'],[32,31,1,0],[0,1,32,0]):
        job.update(active_rows=count,parent_base=base,reference_rows=[0]*count,
                   raw_scores=[ref[0][:] for _ in range(count)])
    execution_text=json.dumps(execution)
    execution_hash=hashlib.sha256(execution_text.encode()).hexdigest()
    inventory.update(schema_version=1,scope='captured_graph_kernel_inventory_not_admission',
        production_admitted=False,kernel_coverage_complete=False,
        device=deepcopy(expected['device']),execution_sha256=execution_hash)
    inventory_text=json.dumps(inventory)
    # Independent literal native enum mapping: FP16=0, ReLU=1.
    dims=dict(state_len=96,num_classes=6,num_pieces=56,max_piece_size=3,seq_len=57,
        padded_seq_len=57,sequence_alignment=1,d_model=256,nhead=8,head_dim=32,
        transformer_layers=4,ff_dim=1024,output_dim=24,dtype=0,activation=1)
    nodes=[]
    for index,node in enumerate(inventory['lanes'][0]['kernels']):
        fid=node['function_id'];arg={0:4,3:2,4:8}.get(fid)
        nodes.append(dict(kernel_index=index,function_id=fid,dims_signature_known=arg is not None,
            dims_index=arg,dims=deepcopy(dims) if arg is not None else None))
    payload=dict(schema_version=1,scope='captured_embedded_dims_not_pointer_members_or_admission',
        production_admitted=False,parameter_values_checked=False,device=deepcopy(expected['device']),
        execution_sha256=execution_hash,inventory_sha256=hashlib.sha256(inventory_text.encode()).hexdigest(),
        lanes=[dict(lane=0,kernels=nodes)])
    if mode=='dtype':nodes[0]['dims']['dtype']=1
    if mode=='state_len':nodes[0]['dims']['state_len']=120
    if mode=='bool_value':nodes[0]['dims']['dtype']=False
    if mode=='device':payload['device']['uuid_hex']='0'*32
    if mode=='inventory_hash':payload['inventory_sha256']='0'*64
    files={'execution.json':execution_text,'raw_scores.json':json.dumps(scores),
        'graph_kernel_inventory.json':inventory_text}
    if mode!='missing':files['graph_embedded_dims.json']=json.dumps(payload)
    code='import pathlib,sys\nout=pathlib.Path(sys.argv[3]);out.mkdir()\n'
    for name,value in files.items():code+=f'(out/{name!r}).write_text({value!r})\n'
    def observe():
        return dict(expected=deepcopy(expected),reference_dir=str(tmp_path),
            reference_scores=deepcopy(ref),require_graph_inventory=True,
            expected_graph_dims_profile=dict(model=model,padded_seq_len=57,ln_shared_bytes=16))
    if mode=='valid':
        receipt=collect_fresh_candidate([sys.executable,'-c',code],tmp_path/'run',5,observe,ref)
        assert receipt.get('graph_embedded_dims_checked') is True
        assert receipt['production_admitted'] is False
    else:
        with pytest.raises(ValueError):
            collect_fresh_candidate([sys.executable,'-c',code],tmp_path/'run',5,observe,ref)
