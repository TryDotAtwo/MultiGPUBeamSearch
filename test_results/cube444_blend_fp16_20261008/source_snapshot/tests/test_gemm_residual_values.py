from copy import deepcopy
import pytest
try:
    from tools.gemm_residual_values import validate_residual_gemm
except ImportError:
    validate_residual_gemm=None


def fixture():
    pointers={name:dict(role=role,offset=0,available_bytes=size) for name,role,size in (
        ('A','lane_context',6656),('B','block3_attn_out_weight',131072),
        ('C','lane_tokens',6656),('D','lane_tokens',6656),
        ('semaphore','null',0),('gather_A','null',0),('gather_B','null',0),('scatter_D','null',0))}
    return dict(problem=dict(m=13,n=256,k=256),pointers=pointers,
        scalars=dict(alpha=1.0,beta=1.0,gemm_k_size=256,lda=256,ldb=256,ldc=256,ldd=256))


def test_cls_attention_out_literal_contract():
    assert callable(validate_residual_gemm), 'independent residual GEMM validator missing'
    node=fixture();node['pointers']['A']['role']='lane_attention'
    node['pointers']['C']['role']=node['pointers']['D']['role']='lane_qkv'
    assert validate_residual_gemm(node,layer=3,rows=13,kind='attn_out',cls=True) is True


@pytest.mark.parametrize('mutation',['weight','input','residual','offset','short','beta',
    'alpha','stride','k_size','rows','bool_dimension','nonnull_aux','missing','extra'])
def test_wrong_residual_gemm_reject(mutation):
    assert callable(validate_residual_gemm), 'independent residual GEMM validator missing'
    n=deepcopy(fixture())
    assert validate_residual_gemm(n,layer=3,rows=13,kind='attn_out',cls=False) is True
    if mutation=='weight':n['pointers']['B']['role']='block2_attn_out_weight'
    elif mutation=='input':n['pointers']['A']['role']='lane_tokens'
    elif mutation=='residual':n['pointers']['C']['role']='lane_context'
    elif mutation=='offset':n['pointers']['B']['offset']=512
    elif mutation=='short':n['pointers']['D']['available_bytes']=1
    elif mutation=='beta':n['scalars']['beta']=0.0
    elif mutation=='alpha':n['scalars']['alpha']=2.0
    elif mutation=='stride':n['scalars']['ldb']=768
    elif mutation=='k_size':n['scalars']['gemm_k_size']=32
    elif mutation=='rows':n['problem']['m']=12
    elif mutation=='bool_dimension':n['problem']['n']=True
    elif mutation=='nonnull_aux':n['pointers']['gather_A']=dict(role='lane_qkv',offset=0,available_bytes=512)
    elif mutation=='missing':del n['scalars']['ldd']
    elif mutation=='extra':n['pointers']['unknown']=dict(role='null',offset=0,available_bytes=0)
    with pytest.raises(ValueError):validate_residual_gemm(n,layer=3,rows=13,kind='attn_out',cls=False)
