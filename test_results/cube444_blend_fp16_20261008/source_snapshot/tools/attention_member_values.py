"""Independent Cube4 FP16 captured FMHA values; no node coverage/admission."""
import math


def validate_attention_params(pointers, scalars, *, batch, seq, cls=False, split=False):
    if (type(batch) is not int or not 1 <= batch <= 65536 or
        type(seq) is not int or seq not in (57,64) or
        type(cls) is not bool or type(split) is not bool or split and not cls):
        raise ValueError('unsupported attention value profile')
    expected = dict(num_heads=8,num_batches=batch,head_dim=32,head_dim_value=32,
        num_queries=1 if cls else seq,num_keys=57,custom_mask_type=0,
        q_strideH=32,k_strideH=32,v_strideH=32,
        q_strideM=256 if split else 768,k_strideM=768,v_strideM=768,
        q_strideB=256 if split else seq*768,k_strideB=seq*768,
        v_strideB=seq*768,o_strideM=256)
    if not isinstance(scalars,dict) or set(scalars)!=set(expected)|{'scale'}:
        raise ValueError('incomplete captured attention scalars')
    for name,value in expected.items():
        if type(scalars[name]) is not int or scalars[name]!=value:
            raise ValueError('wrong captured attention scalar: '+name)
    scale=scalars['scale']
    if type(scale) is not float or not math.isfinite(scale) or abs(scale-0.1767766922712326)>1e-9:
        raise ValueError('wrong captured attention scale')
    total=batch*seq*1536
    endpoints = {
        'query_ptr':('lane_ff_hidden',0,batch*512) if split else ('lane_qkv',0,total),
        'key_ptr':('lane_qkv',512,total-512),
        'value_ptr':('lane_qkv',1024,total-1024),
        'output_ptr':('lane_attention' if cls else 'lane_context',0,batch*(1 if cls else seq)*512),
        'output_accum_ptr':('null',0,0),'logsumexp_ptr':('null',0,0)}
    if not isinstance(pointers,dict) or set(pointers)!=set(endpoints):
        raise ValueError('incomplete captured attention pointers')
    for name,(role,offset,minimum) in endpoints.items():
        p=pointers[name]
        if (not isinstance(p,dict) or set(p)!={'role','offset','available_bytes'} or
            p['role']!=role or type(p['offset']) is not int or p['offset']!=offset or
            type(p['available_bytes']) is not int or not minimum<=p['available_bytes']<=2**64-1 or
            role=='null' and p['available_bytes']!=0):
            raise ValueError('wrong captured attention pointer: '+name)
    return True
