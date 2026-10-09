"""Padding writes must bind a known role, full allocation extent and launch grid."""
from copy import deepcopy
import pytest
from test_graph_pointer_contract import fixture as pointer_fixture
from tools.graph_pointer_contract import validate_graph_pointer_roles

def fixture():
    value,model,execution=pointer_fixture();execution['expected_padded_seq_len']=64
    nodes=value['lanes'][0]['kernels']
    for node in nodes:
        for pointer in node['pointers']:
            if pointer['available_bytes']==58368:pointer['available_bytes']=65536
    name='stream1_transformer_zero_padded_rows_kernel'
    symbol=f'_ZN4beam{len(name)}{name}EP6__halfNS_22Stream1TransformerDimsEjj'
    for role in ['lane_tokens']*4+['lane_context']*3:
        nodes.append(dict(kernel_index=len(nodes),mangled_name=symbol,pointer_signature_known=True,
            pointers=[dict(index=0,role=role,offset=0,available_bytes=65536)]))
    return value,model,execution

def check(value,model,execution):return validate_graph_pointer_roles(value,model,execution,16)

def test_seven_padding_writes_bind_independent_roles():
    value,model,execution=fixture();result=check(value,model,execution)
    assert result['checked_pointer_kernel_nodes']==len(value['lanes'][0]['kernels'])
    assert result['production_admitted'] is False

@pytest.mark.parametrize('mutation',['null','role','offset','short','index','bool_extent','opaque','missing','wrong_multiplicity'])
def test_bad_padding_pointer_is_rejected(mutation):
    value,model,execution=fixture();nodes=value['lanes'][0]['kernels'];pointer=nodes[-1]['pointers'][0]
    if mutation=='null':pointer.update(role='null',available_bytes=0)
    elif mutation=='role':pointer['role']='lane_qkv'
    elif mutation=='offset':pointer['offset']=2
    elif mutation=='short':pointer['available_bytes']=65535
    elif mutation=='index':pointer['index']=1
    elif mutation=='bool_extent':pointer['available_bytes']=True
    elif mutation=='opaque':nodes[-1].update(pointer_signature_known=False,pointers=[])
    elif mutation=='missing':nodes.pop()
    else:pointer['role']='lane_tokens'
    with pytest.raises(ValueError):check(value,model,execution)
