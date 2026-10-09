"""Independent residual GEMM member values, not graph order/coverage/admission."""


def validate_residual_gemm(node, *, layer, rows, kind, cls=False):
    if (type(layer) is not int or not 0<=layer<4 or type(rows) is not int or
        not 0<rows<=65536*64 or kind not in ('attn_out','ff2') or
        type(cls) is not bool or cls and layer!=3):
        raise ValueError('unsupported residual GEMM profile')
    k=256 if kind=='attn_out' else 1024
    problem=dict(m=rows,n=256,k=k)
    actual=node.get('problem') if isinstance(node,dict) else None
    if (not isinstance(actual,dict) or set(actual)!=set(problem) or
        any(type(actual[name]) is not int or actual[name]!=value for name,value in problem.items())):
        raise ValueError('wrong residual GEMM problem')
    scalars=dict(alpha=1.0,beta=1.0,gemm_k_size=k,lda=k,ldb=256,ldc=256,ldd=256)
    actual=node.get('scalars')
    if (not isinstance(actual,dict) or set(actual)!=set(scalars) or
        any(type(actual[name]) is not type(value) or actual[name]!=value for name,value in scalars.items())):
        raise ValueError('wrong residual GEMM scalar')
    input_role=('lane_attention' if cls else 'lane_context') if kind=='attn_out' else 'lane_ff_hidden'
    output_role='lane_qkv' if cls else 'lane_tokens'
    wanted={'A':(input_role,rows*k*2),
        'B':(f'block{layer}_{kind}_weight',k*256*2),
        'C':(output_role,rows*256*2),'D':(output_role,rows*256*2),
        'semaphore':('null',0),'gather_A':('null',0),
        'gather_B':('null',0),'scatter_D':('null',0)}
    pointers=node.get('pointers')
    if not isinstance(pointers,dict) or set(pointers)!=set(wanted):
        raise ValueError('incomplete residual GEMM pointers')
    for name,(role,minimum) in wanted.items():
        p=pointers[name]
        if (not isinstance(p,dict) or set(p)!={'role','offset','available_bytes'} or
            p['role']!=role or type(p['offset']) is not int or p['offset']!=0 or
            type(p['available_bytes']) is not int or not minimum<=p['available_bytes']<=2**64-1 or
            role=='null' and p['available_bytes']!=0):
            raise ValueError('wrong residual GEMM endpoint: '+name)
    return True
