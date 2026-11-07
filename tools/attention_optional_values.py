"""Source-pinned default FMHA Params checks; not full kernel admission."""
import math

POINTERS = ('attn_bias_ptr', 'seqstart_q_ptr', 'seqstart_k_ptr', 'seqlen_k_ptr')
SCALARS = dict(causal_diagonal_offset=0, num_keys_absolute=0, bias_strideM=0,
               bias_strideH=0, bias_strideB=0, use_dropout=False,
               dropout_batch_head_rng_offset=0, dropout_prob=0.0)

def validate_attention_optional_members(node):
    pointers=node.get('optional_pointers')
    scalars=node.get('optional_scalars')
    if not isinstance(pointers,dict) or set(pointers)!=set(POINTERS):
        raise ValueError('incomplete optional attention pointers')
    for name in POINTERS:
        p=pointers[name]
        if (not isinstance(p,dict) or set(p)!={'role','offset','available_bytes'} or
            p['role']!='null' or type(p['offset']) is not int or p['offset']!=0 or
            type(p['available_bytes']) is not int or p['available_bytes']!=0):
            raise ValueError('nondefault optional attention pointer: '+name)
    if not isinstance(scalars,dict) or set(scalars)!=set(SCALARS):
        raise ValueError('incomplete optional attention scalars')
    for name,expected in SCALARS.items():
        value=scalars[name]
        if type(value) is not type(expected) or value!=expected:
            raise ValueError('nondefault optional attention scalar: '+name)
        if type(value) is float and not math.isfinite(value):
            raise ValueError('nonfinite optional attention scalar')
    if node.get('pytorch_rng_state_present') is not False:
        raise ValueError('unsupported PyTorch RNG Params layout')
    return True
