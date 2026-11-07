"""Classic unsplit QKV/FF1 bias members; row iterator Params/order not attested."""


def validate_bias_gemm(node, *, layer, rows, kind):
    if type(layer) is not int or not 0<=layer<4 or type(rows) is not int or not 0<rows<=65536*64 or kind not in ('qkv','ff1'):
        raise ValueError('unsupported bias GEMM profile')
    n=768 if kind=='qkv' else 1024
    family='attn_qkv' if kind=='qkv' else 'ff1'
    output='lane_qkv' if kind=='qkv' else 'lane_ff_hidden'
    problem=dict(m=rows,n=n,k=256)
    actual=node.get('problem') if isinstance(node,dict) else None
    if not isinstance(actual,dict) or set(actual)!=set(problem) or any(type(actual[k]) is not int or actual[k]!=v for k,v in problem.items()):
        raise ValueError('wrong bias GEMM problem')
    scalars=dict(alpha=1.0,beta=0.0,batch_stride_A=rows*256,batch_stride_B=256*n,
        batch_stride_C=rows*n,batch_stride_Vector=0,batch_stride_Tensor=0,ldr=0)
    actual=node.get('scalars')
    if not isinstance(actual,dict) or set(actual)!=set(scalars) or any(type(actual[k]) is not type(v) or actual[k]!=v for k,v in scalars.items()):
        raise ValueError('wrong bias GEMM scalar')
    wanted={'A':('lane_context',rows*256*2),'B':(f'block{layer}_{family}_weight',256*n*2),
        'C':(output,rows*n*2),'D':(output,rows*n*2),
        'Vector':(f'block{layer}_{family}_bias',n*2),'Tensor':('null',0)}
    pointers=node.get('pointers')
    if not isinstance(pointers,dict) or set(pointers)!=set(wanted):raise ValueError('incomplete bias GEMM pointers')
    for name,(role,minimum) in wanted.items():
        p=pointers[name]
        if (not isinstance(p,dict) or set(p)!={'role','offset','available_bytes'} or
            p['role']!=role or type(p['offset']) is not int or p['offset']!=0 or
            type(p['available_bytes']) is not int or not minimum<=p['available_bytes']<=2**64-1 or
            role=='null' and p['available_bytes']!=0):
            raise ValueError('wrong bias GEMM endpoint: '+name)
    return True
