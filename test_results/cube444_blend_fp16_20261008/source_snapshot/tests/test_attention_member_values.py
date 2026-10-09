from copy import deepcopy
import pytest
try:
    from tools.attention_member_values import validate_attention_params
except ImportError:
    validate_attention_params = None


def fixture():
    pointers = {
        'query_ptr':dict(role='lane_qkv',offset=0,available_bytes=19968),
        'key_ptr':dict(role='lane_qkv',offset=512,available_bytes=19456),
        'value_ptr':dict(role='lane_qkv',offset=1024,available_bytes=18944),
        'output_ptr':dict(role='lane_context',offset=0,available_bytes=6656),
        'output_accum_ptr':dict(role='null',offset=0,available_bytes=0),
        'logsumexp_ptr':dict(role='null',offset=0,available_bytes=0)}
    scalars = dict(scale=0.1767766922712326,num_heads=8,num_batches=13,
        head_dim=32,head_dim_value=32,num_queries=1,num_keys=57,custom_mask_type=0,
        q_strideH=32,k_strideH=32,v_strideH=32,q_strideM=768,k_strideM=768,
        v_strideM=768,q_strideB=43776,k_strideB=43776,v_strideB=43776,o_strideM=256)
    # Unsplitted CLS queries still stride across each whole padded sequence.
    for name in ('query_ptr','key_ptr','value_ptr'):
        pointers[name]['available_bytes']=1138176-pointers[name]['offset']
    pointers['output_ptr']['role']='lane_attention'
    return pointers,scalars


def test_known_cls_params_independent_values():
    assert callable(validate_attention_params), 'captured attention value validator missing'
    assert validate_attention_params(*fixture(),batch=13,seq=57,cls=True,split=False) is True


@pytest.mark.parametrize('mutation',['scale','stride','batch','queries','mask',
    'bool_head','qk_swap','short','offset','null_output','nonnull_lse','missing','extra'])
def test_invalid_params_reject(mutation):
    assert callable(validate_attention_params), 'captured attention value validator missing'
    p,s=deepcopy(fixture())
    if mutation=='scale':s['scale']=1.0
    elif mutation=='stride':s['k_strideB']=768
    elif mutation=='batch':s['num_batches']=12
    elif mutation=='queries':s['num_queries']=57
    elif mutation=='mask':s['custom_mask_type']=1
    elif mutation=='bool_head':s['num_heads']=True
    elif mutation=='qk_swap':p['query_ptr'],p['key_ptr']=p['key_ptr'],p['query_ptr']
    elif mutation=='short':p['value_ptr']['available_bytes']=1
    elif mutation=='offset':p['output_ptr']['offset']=512
    elif mutation=='null_output':p['output_ptr']=dict(role='null',offset=0,available_bytes=0)
    elif mutation=='nonnull_lse':p['logsumexp_ptr']=dict(role='lane_context',offset=0,available_bytes=512)
    elif mutation=='missing':del s['q_strideM']
    elif mutation=='extra':s['unvalidated']=0
    with pytest.raises(ValueError):
        validate_attention_params(p,s,batch=13,seq=57,cls=True,split=False)
