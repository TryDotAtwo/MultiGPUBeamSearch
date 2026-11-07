"""Bounded structural validation only; symbols/profile/replay need separate proof."""
import math


def validate_graph_inventory_structure(inventory, *, lanes, sm):
    def integer(value, lower, upper):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError('invalid graph inventory integer')
        return value

    integer(lanes,1,65536)
    integer(sm,1,999)
    if not isinstance(inventory,dict):
        raise ValueError('graph inventory must be an object')
    functions=inventory.get('functions')
    captures=inventory.get('lanes')
    if not isinstance(functions,list) or not 1 <= len(functions) <= 65536:
        raise ValueError('missing or excessive graph functions')
    if not isinstance(captures,list) or len(captures)!=lanes:
        raise ValueError('graph lane count mismatch')
    attrs_keys={'binary_version','ptx_version','max_threads_per_block','num_regs',
                'static_shared_bytes','local_bytes','constant_bytes','max_dynamic_shared_bytes'}
    for function in functions:
        if not isinstance(function,dict) or set(function)!={'mangled_name','attributes'}:
            raise ValueError('malformed graph function')
        symbol=function['mangled_name']
        if (not isinstance(symbol,str) or not symbol.startswith('_Z') or
                not 3 <= len(symbol) <= 65536 or any(ord(c)<33 or ord(c)>126 for c in symbol)):
            raise ValueError('invalid graph symbol')
        # IDs represent distinct observed function pointers. Separate host
        # instantiations can expose the same CUDA mangled name (actual Sm75).
        # Keep every ID and every launch; never coalesce by descriptive name.
        attrs=function['attributes']
        if not isinstance(attrs,dict) or set(attrs)!=attrs_keys:
            raise ValueError('malformed graph attributes')
        integer(attrs['binary_version'],1,sm)
        integer(attrs['ptx_version'],1,sm)
        integer(attrs['max_threads_per_block'],1,1024)
        integer(attrs['num_regs'],1,255)
        for key in ('static_shared_bytes','local_bytes','constant_bytes','max_dynamic_shared_bytes'):
            integer(attrs[key],0,2**32-1)
    seen_lanes=set()
    used=set()
    total=0
    for capture in captures:
        if not isinstance(capture,dict) or set(capture)!={'lane','node_count','node_types','kernels'}:
            raise ValueError('malformed graph capture')
        lane=integer(capture['lane'],0,lanes-1)
        if lane in seen_lanes: raise ValueError('duplicate graph lane')
        seen_lanes.add(lane)
        node_count=integer(capture['node_count'],1,65536)
        types=capture['node_types']
        if not isinstance(types,dict) or set(types)!={'kernel','memcpy','memset'}:
            raise ValueError('unknown graph node type')
        if sum(integer(v,0,65536) for v in types.values())!=node_count:
            raise ValueError('graph node count mismatch')
        kernels=capture['kernels']
        if not isinstance(kernels,list) or not kernels or len(kernels)!=types['kernel']:
            raise ValueError('graph kernel count mismatch')
        for kernel in kernels:
            if not isinstance(kernel,dict) or set(kernel)!={'function_id','grid','block','dynamic_shared_bytes'}:
                raise ValueError('malformed graph kernel')
            fid=integer(kernel['function_id'],0,len(functions)-1)
            used.add(fid)
            for key,limits in (('grid',(2**31-1,65535,65535)),('block',(1024,1024,64))):
                dimensions=kernel[key]
                if not isinstance(dimensions,list) or len(dimensions)!=3:
                    raise ValueError('malformed graph dimensions')
                for dimension,limit in zip(dimensions,limits): integer(dimension,1,limit)
            attrs=functions[fid]['attributes']
            if math.prod(kernel['block'])>attrs['max_threads_per_block']:
                raise ValueError('graph block exceeds function limit')
            integer(kernel['dynamic_shared_bytes'],0,attrs['max_dynamic_shared_bytes'])
        total+=len(kernels)
    if used!=set(range(len(functions))):
        raise ValueError('unused graph function metadata')
    return dict(functions=len(functions),kernel_nodes=total,structure_checked=True,
                production_admitted=False,kernel_coverage_complete=False)
