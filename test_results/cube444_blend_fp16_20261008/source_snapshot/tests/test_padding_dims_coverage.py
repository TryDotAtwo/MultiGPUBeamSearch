"""Consumed padding dimensions cannot remain opaque in canonical seq64 graphs."""
from copy import deepcopy
import pytest
from test_graph_scalar_contract import fixture as graph_fixture
from tools.graph_dims_contract import validate_graph_embedded_dims

def fixture():
    _,inventory,model,execution=graph_fixture()
    execution['expected_padded_seq_len']=64
    name='stream1_transformer_zero_padded_rows_kernel'
    symbol=f'_ZN4beam{len(name)}{name}EP6__halfNS_22Stream1TransformerDimsEjj'
    fid=len(inventory['functions'])
    inventory['functions'].append(dict(mangled_name=symbol,attributes=deepcopy(inventory['functions'][0]['attributes'])))
    lane=inventory['lanes'][0]
    # Seven padding calls per chunk: input + attention/FFN outputs in first3layers.
    for _ in range(21):
        lane['kernels'].append(dict(function_id=fid,grid=[1,1,1],block=[256,1,1],dynamic_shared_bytes=0))
    lane['node_count']=len(lane['kernels']);lane['node_types']['kernel']=lane['node_count']
    dims=dict(state_len=96,num_classes=6,num_pieces=56,max_piece_size=3,seq_len=57,
              padded_seq_len=64,sequence_alignment=16,d_model=256,nhead=8,head_dim=32,
              transformer_layers=4,ff_dim=1024,output_dim=24,dtype=0,activation=1)
    nodes=[]
    for index,node in enumerate(lane['kernels']):
        arg={0:4,3:2,4:8,fid:1}.get(node['function_id'])
        nodes.append(dict(kernel_index=index,function_id=node['function_id'],dims_signature_known=arg is not None,
                          dims_index=arg,dims=deepcopy(dims) if arg is not None else None))
    payload=dict(schema_version=1,scope='captured_embedded_dims_not_pointer_members_or_admission',
                 production_admitted=False,parameter_values_checked=False,lanes=[dict(lane=0,kernels=nodes)])
    return payload,inventory,model,execution

def test_all_padding_calls_require_dimensions():
    result=validate_graph_embedded_dims(*fixture(),16)
    assert result['checked_dimension_kernel_nodes']==30
    assert result['production_admitted'] is False

@pytest.mark.parametrize('field',['state_len','num_classes','num_pieces','max_piece_size','seq_len',
    'padded_seq_len','sequence_alignment','d_model','nhead','head_dim','transformer_layers','ff_dim',
    'output_dim','dtype','activation'])
def test_padding_dimension_value_mutation_rejected(field):
    payload,inventory,model,execution=fixture()
    payload['lanes'][0]['kernels'][-1]['dims'][field]+=1
    with pytest.raises(ValueError):validate_graph_embedded_dims(payload,inventory,model,execution,16)

def test_padding_dimensions_cannot_be_opaque():
    payload,inventory,model,execution=fixture()
    payload['lanes'][0]['kernels'][-1].update(dims_signature_known=False,dims_index=None,dims=None)
    with pytest.raises(ValueError):validate_graph_embedded_dims(payload,inventory,model,execution,16)
