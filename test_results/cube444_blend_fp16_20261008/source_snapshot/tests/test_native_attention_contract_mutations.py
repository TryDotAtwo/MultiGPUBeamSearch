"""Mutation of actual captured report, not CUDA argument mutation or admission."""
import json
import os
from pathlib import Path
import pytest
from tools.graph_attention_members_contract import validate_graph_attention_members


@pytest.mark.parametrize('mutation',['valid','missing_lane','duplicate_lane','missing_node',
    'duplicate_index','wrong_function','unknown_promoted','bool_index','wrong_param',
    'stride','short','role','scale','scope','admission'])
def test_actual_attention_report_contract(mutation):
    selected=os.environ.get('BEAM_TEST_ATTENTION_SCORE_OUTPUT')
    if not selected:pytest.skip('explicit remote captured artifact required')
    root=Path(selected)
    def read(name):return json.loads((root/name).read_bytes())
    payload=read('graph_attention_members.json');inventory=read('graph_kernel_inventory.json')
    execution=read('execution.json')
    execution['expected_padded_seq_len']=int(os.environ['BEAM_TEST_ATTENTION_PADDED_SEQ'])
    model=dict(dtype='fp16',num_layers=4,nhead=8,head_dim=32,seq_len=57,d_model=256)
    lane=payload['lanes'][0]
    node=next(n for n in lane['kernels'] if n['attention_signature_known'])
    if mutation=='missing_lane':payload['lanes'].pop()
    elif mutation=='duplicate_lane':payload['lanes'].append(lane)
    elif mutation=='missing_node':lane['kernels'].pop()
    elif mutation=='duplicate_index':node['kernel_index']=lane['kernels'][0]['kernel_index']
    elif mutation=='wrong_function':node['function_id']=99999
    elif mutation=='unknown_promoted':lane['kernels'][0]['attention_signature_known']=True
    elif mutation=='bool_index':node['kernel_index']=True
    elif mutation=='wrong_param':node['parameter_index']=1
    elif mutation=='stride':node['scalars']['v_strideB']=1
    elif mutation=='short':node['pointers']['query_ptr']['available_bytes']=1
    elif mutation=='role':node['pointers']['query_ptr']['role']='lane_context'
    elif mutation=='scale':node['scalars']['scale']=1.0
    elif mutation=='scope':payload['scope']='complete'
    elif mutation=='admission':payload['production_admitted']=True
    if mutation=='valid':assert validate_graph_attention_members(payload,inventory,model,execution) is True
    else:
        with pytest.raises(ValueError):validate_graph_attention_members(payload,inventory,model,execution)
