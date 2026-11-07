from copy import deepcopy
import pytest
try:
    from tools.gemm_head_values import validate_head_gemm
except ImportError:
    validate_head_gemm=None


def fixture():
    return dict(problem=dict(m=6,n=24,k=256),
        scalars=dict(alpha=1.0,beta=0.0,gemm_k_size=256,lda=256,ldb=24,ldc=24,ldd=24),
        pointers={name:dict(role=role,offset=0,available_bytes=size) for name,role,size in (
            ('A','lane_context',3072),('B','output_weight',12288),
            ('C','lane_logits',288),('D','lane_logits',288),
            ('semaphore','null',0),('gather_A','null',0),('gather_B','null',0),('scatter_D','null',0))})


def test_literal_head():
    assert callable(validate_head_gemm),'head validator missing'
    assert validate_head_gemm(fixture(),batch=6)


@pytest.mark.parametrize('mutation',['weight','input','output','short','stride','beta','shape','aux'])
def test_wrong_head_rejects(mutation):
    assert callable(validate_head_gemm),'head validator missing'
    node=deepcopy(fixture())
    assert validate_head_gemm(node,batch=6)
    if mutation=='weight':node['pointers']['B']['role']='block0_attn_out_weight'
    elif mutation=='input':node['pointers']['A']['role']='lane_tokens'
    elif mutation=='output':node['pointers']['D']['role']='lane_context'
    elif mutation=='short':node['pointers']['B']['available_bytes']=12287
    elif mutation=='stride':node['scalars']['ldb']=256
    elif mutation=='beta':node['scalars']['beta']=1.0
    elif mutation=='shape':node['problem']['n']=256
    elif mutation=='aux':node['pointers']['gather_A']=dict(role='lane_context',offset=0,available_bytes=1)
    with pytest.raises(ValueError):validate_head_gemm(node,batch=6)
