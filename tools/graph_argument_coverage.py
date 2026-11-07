"""Compose already validated receipts into exhaustive argument/member accounting.

This checks coverage joins, not values; callers must first run independent value,
role, extent, source, geometry and execution validators. Never grants admission.
"""
from tools.cuda_kernel_symbol import decode_kernel_symbols
from tools.graph_parameter_layout import PATTERNS
from tools.public_parameter_storage import reference
from tools.network_member_consumption import CONSUMED

def indexed(payload,inventory):
    captures={lane['lane']:lane['kernels'] for lane in inventory['lanes']};result={}
    for lane in payload['lanes']:
        lid=lane['lane']
        if type(lid) is not int or lid not in captures or lid in result:raise ValueError('coverage lane mismatch')
        nodes={}
        for node in lane['kernels']:
            i=node['kernel_index']
            if type(i) is not int or not 0<=i<len(captures[lid]) or i in nodes:raise ValueError('coverage node mismatch')
            if 'function_id' in node and (type(node['function_id']) is not int or node['function_id']!=captures[lid][i]['function_id']):
                raise ValueError('coverage function mismatch')
            nodes[i]=node
        if set(nodes)!=set(range(len(captures[lid]))):raise ValueError('incomplete coverage receipt nodes')
        result[lid]=nodes
    if set(result)!=set(captures):raise ValueError('incomplete coverage receipt lanes')
    return result

def unique_indices(values):
    indices=[v['index'] for v in values]
    if any(type(i) is not int for i in indices) or len(set(indices))!=len(indices):raise ValueError('invalid covered indices')
    return set(indices)

def member_paths(node,broadcast):
    paths={}
    def add(name,*path):paths[name]=[path]
    for name in ('problem_size','grid_tiled_shape'):
        prefix=('problem',) if name=='problem_size' else ('launch_scalars','grid_tiled_shape')
        paths[name]=[(*prefix,k) for k in ('m','n','k')]
    for name in ('swizzle_log_tile','gemm_k_size','semaphore'):
        add(name,'launch_scalars','semaphore_null' if name=='semaphore' else name)
    for key in ('alpha','beta'):
        add('output_op.'+key,'scalars',key)
        paths['output_op.'+key+'_ptr']=[('epilogue_pointers',key,k) for k in ('role','offset','available_bytes')]
        if not broadcast:paths['output_op.'+key+'_ptr_array']=[('epilogue_pointers',key+'_array',k) for k in ('role','offset','available_bytes')]
    for key in ('A','B'):
        paths['params_'+key]=[('mainloop_iterators',key,k) for k in ('stride','increment_strided','increment_next','advance')]
    epi=('stride','increment_row','increment_group','increment_cluster','advance_row','advance_group','advance_cluster','advance_tile')
    for key in ('C','D'):paths['params_'+key]=[('epilogue_iterators',key,k) for k in epi]
    if broadcast:
        paths['params_Tensor']=[('tensor_iterator',k) for k in epi]
        paths['output_op.elementwise']=[] # separately pinned exact EmptyArguments exclusion
        for name in ('mode','batch_count','batch_stride_D'):add(name,'launch_scalars',name)
        for key in ('A','B','C','D','Vector','Tensor'):
            paths['ptr_'+key]=[('pointers',key,k) for k in ('role','offset','available_bytes')]
        for name in ('batch_stride_A','batch_stride_B','batch_stride_C','batch_stride_Vector','batch_stride_Tensor','ldr'):
            add(name,'scalars',name)
    else:
        for key in ('A','B','C','D'):
            paths['ref_'+key]=[('pointers',key,k) for k in ('role','offset','available_bytes')]+[('scalars','ld'+key.lower())]
        for name,key in (('gather_A_indices','gather_A'),('gather_B_indices','gather_B'),('scatter_D_indices','scatter_D')):
            paths[name]=[('pointers',key,k) for k in ('role','offset','available_bytes')]
    for name,items in paths.items():
        for path in items:
            value=node
            for key in path:
                if not isinstance(value,dict) or key not in value:raise ValueError('uncovered public GEMM member: '+name)
                value=value[key]
    return set(paths)

def validate_argument_coverage(inventory,pointers,scalars,dims,network,gemm,attention,*,empty_arguments_checked):
    tables=reference();descs=decode_kernel_symbols([f['mangled_name'] for f in inventory['functions']])
    receipts=[indexed(p,inventory) for p in (pointers,scalars,dims,network,gemm,attention)]
    counts={'non_gemm_nodes':0,'gemm_nodes':0,'attention_nodes':0,'covered_argument_slots':0}
    for lane in inventory['lanes']:
        lid=lane['lane']
        for i,native in enumerate(lane['kernels']):
            family=descs[native['function_id']]['family'];p,s,d,n,g,a=[r[lid][i] for r in receipts]
            if family in PATTERNS:
                pattern=PATTERNS[family]
                if unique_indices(p['pointers'])!={j for j,t in enumerate(pattern) if t=='pointer'}:
                    raise ValueError('uncovered top-level pointer argument')
                if unique_indices(s['u32_values'])!={j for j,t in enumerate(pattern) if t=='u32'}:
                    raise ValueError('uncovered top-level scalar argument')
                for j,t in enumerate(pattern):
                    if t in ('dims','network') and (d.get('dims_signature_known') is not True or type(d.get('dims_index')) is not int or d['dims_index']!=j):
                        raise ValueError('uncovered dimension-containing argument')
                    if t=='network' and (n.get('network_signature_known') is not True or type(n.get('network_index')) is not int or n['network_index']!=j):
                        raise ValueError('uncovered network pointer members')
                    if t=='network':
                        names=[m['member'] for m in n['members']]
                        if len(names)!=len(set(names)) or set(names)!=CONSUMED[family]:
                            raise ValueError('uncovered consumed NetworkView member')
                counts['non_gemm_nodes']+=1;counts['covered_argument_slots']+=len(pattern)
            elif family=='attention':
                if a.get('attention_signature_known') is not True or type(a.get('parameter_index')) is not int or a['parameter_index']!=0:
                    raise ValueError('uncovered attention Params argument')
                names=[]
                for section in ('pointers','scalars','optional_pointers','optional_scalars'):names.extend(a[section])
                expected={f['field'] for f in tables['attention']['entries'][0]['fields']}
                if len(names)!=len(set(names)) or set(names)!=expected:raise ValueError('uncovered/duplicate attention public member')
                counts['attention_nodes']+=1;counts['covered_argument_slots']+=1
            elif 'stages' in descs[native['function_id']]:
                if g.get('gemm_signature_known') is not True or type(g.get('parameter_index')) is not int or g['parameter_index']!=0:
                    raise ValueError('uncovered GEMM Params argument')
                broadcast=family in ('gemm_bias','gemm_relu')
                if broadcast and empty_arguments_checked is not True:raise ValueError('missing empty activation exclusion')
                row=tables['gemm']['entries'][1 if broadcast else 0]
                if member_paths(g,broadcast)!={f['field'] for f in row['fields']}:
                    raise ValueError('public GEMM field coverage/reference disagreement')
                counts['gemm_nodes']+=1;counts['covered_argument_slots']+=1
            else:raise ValueError('unknown argument coverage family')
    if not all(counts[k]>0 for k in ('non_gemm_nodes','gemm_nodes','attention_nodes')):raise ValueError('incomplete composed graph families')
    return dict(scope='composed_argument_slots_and_public_members_not_full_admission',**counts,
        production_admitted=False,parameter_values_checked=False)
