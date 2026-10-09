"""Bounded classic FP16 head members, not dependency or iterator admission."""


def validate_head_gemm(node, *, batch):
    if type(batch) is not int or not 0<batch<=65536:
        raise ValueError('unsupported head batch')
    if not isinstance(node,dict):raise ValueError('missing head node')
    for name,wanted in (
        ('problem',dict(m=batch,n=24,k=256)),
        ('scalars',dict(alpha=1.0,beta=0.0,gemm_k_size=256,lda=256,ldb=24,ldc=24,ldd=24))):
        actual=node.get(name)
        if not isinstance(actual,dict) or set(actual)!=set(wanted) or any(type(actual[k]) is not type(v) or actual[k]!=v for k,v in wanted.items()):
            raise ValueError('wrong head '+name)
    wanted={'A':('lane_context',batch*256*2),'B':('output_weight',12288),
        'C':('lane_logits',batch*24*2),'D':('lane_logits',batch*24*2),
        'semaphore':('null',0),'gather_A':('null',0),'gather_B':('null',0),'scatter_D':('null',0)}
    actual=node.get('pointers')
    if not isinstance(actual,dict) or set(actual)!=set(wanted):raise ValueError('incomplete head pointers')
    for name,(role,minimum) in wanted.items():
        p=actual[name]
        if (not isinstance(p,dict) or set(p)!={'role','offset','available_bytes'} or
            p['role']!=role or type(p['offset']) is not int or p['offset']!=0 or
            type(p['available_bytes']) is not int or not minimum<=p['available_bytes']<=2**64-1 or
            role=='null' and p['available_bytes']!=0):
            raise ValueError('wrong head endpoint: '+name)
    return True
