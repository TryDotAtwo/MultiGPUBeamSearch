from copy import deepcopy
import pytest
try:
    from tools.gemm_bias_values import validate_bias_gemm
except ImportError:
    validate_bias_gemm=None


def fixture():
    return dict(problem=dict(m=13,n=768,k=256),
        pointers={name:dict(role=role,offset=0,available_bytes=size) for name,role,size in (
            ('A','lane_context',6656),('B','block0_attn_qkv_weight',393216),
            ('C','lane_qkv',19968),('D','lane_qkv',19968),
            ('Vector','block0_attn_qkv_bias',1536),('Tensor','null',0))},
        scalars=dict(alpha=1.0,beta=0.0,batch_stride_A=3328,batch_stride_B=196608,
            batch_stride_C=9984,batch_stride_Vector=0,batch_stride_Tensor=0,ldr=0))


def test_qkv_literal_bias_contract():
    assert callable(validate_bias_gemm), 'independent bias GEMM validator missing'
    assert validate_bias_gemm(fixture(),layer=0,rows=13,kind='qkv') is True


def test_ff1_literal_bias_contract():
    node=dict(problem=dict(m=13,n=1024,k=256),
        pointers={name:dict(role=role,offset=0,available_bytes=size) for name,role,size in (
            ('A','lane_context',6656),('B','block2_ff1_weight',524288),
            ('C','lane_ff_hidden',26624),('D','lane_ff_hidden',26624),
            ('Vector','block2_ff1_bias',2048),('Tensor','null',0))},
        scalars=dict(alpha=1.0,beta=0.0,batch_stride_A=3328,batch_stride_B=262144,
            batch_stride_C=13312,batch_stride_Vector=0,batch_stride_Tensor=0,ldr=0))
    assert validate_bias_gemm(node,layer=2,rows=13,kind='ff1') is True


@pytest.mark.parametrize('mutation',['weight','bias','output','input','offset','short',
    'beta','alpha','batch_stride','vector_stride','tensor','shape','bool_rows','missing','extra'])
def test_wrong_bias_gemm_reject(mutation):
    assert callable(validate_bias_gemm), 'independent bias GEMM validator missing'
    n=deepcopy(fixture())
    assert validate_bias_gemm(n,layer=0,rows=13,kind='qkv') is True
    if mutation=='weight':n['pointers']['B']['role']='block1_attn_qkv_weight'
    elif mutation=='bias':n['pointers']['Vector']['role']='block1_attn_qkv_bias'
    elif mutation=='output':n['pointers']['D']['role']='lane_context'
    elif mutation=='input':n['pointers']['A']['role']='lane_tokens'
    elif mutation=='offset':n['pointers']['B']['offset']=512
    elif mutation=='short':n['pointers']['Vector']['available_bytes']=512
    elif mutation=='beta':n['scalars']['beta']=1.0
    elif mutation=='alpha':n['scalars']['alpha']=0.0
    elif mutation=='batch_stride':n['scalars']['batch_stride_B']=65536
    elif mutation=='vector_stride':n['scalars']['batch_stride_Vector']=1
    elif mutation=='tensor':n['pointers']['Tensor']=dict(role='lane_qkv',offset=0,available_bytes=1)
    elif mutation=='shape':n['problem']['n']=256
    elif mutation=='bool_rows':n['problem']['m']=True
    elif mutation=='missing':del n['pointers']['Vector']
    elif mutation=='extra':n['scalars']['unknown']=0
    with pytest.raises(ValueError):validate_bias_gemm(n,layer=0,rows=13,kind='qkv')
