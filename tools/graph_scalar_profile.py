"""Independent non-GEMM uint32 values; pointers/structs/admission are separate."""
from collections import Counter
import json
from tools.graph_non_gemm_profile import expected_graph_non_gemms
from tools.cuda_kernel_symbol import decode_kernel_symbols
from tools.graph_inventory_contract import validate_graph_inventory_structure

SCALAR_INDICES={'fused_input':(6,7),'input_graph':(6,7),'ln':(4,5),
    'bias_ln':(5,6),'gather_cls':(3,4),'cls_ln':(6,),
    'padding':(2,3),'padding_legacy':(2,),'quantize_graph':(5,6,7)}


def expected_graph_non_gemm_scalars(model,execution,ln_shared_bytes):
    # Validate the same supported policies/model/profile before deriving values.
    expected_graph_non_gemms(model,execution,ln_shared_bytes)
    outer=execution['outer_microbatch'];inner=execution['transformer_microbatch']
    seq=execution['expected_padded_seq_len'];result=Counter()
    for offset in range(0,outer,inner):
        batch=min(inner,outer-offset)
        single=dict(execution,outer_microbatch=batch,transformer_microbatch=batch)
        geometry=expected_graph_non_gemms(model,single,ln_shared_bytes)
        for (description,grid,block,shared),count in geometry.items():
            family=json.loads(description)['family']
            if family in ('fused_input','input_graph'):values=((6,batch),(7,offset))
            elif family=='ln':values=((4,grid[0]),(5,0))
            elif family=='bias_ln':values=((5,grid[0]),(6,0))
            elif family=='gather_cls':values=((3,batch),(4,seq))
            elif family=='cls_ln':values=((6,batch),)
            elif family=='padding':values=((2,batch),(3,seq-57))
            elif family=='padding_legacy':values=((2,batch),)
            elif family=='quantize_graph':values=((5,batch),(6,outer),(7,offset))
            else:raise ValueError('unsupported scalar signature')
            result[(family,values)]+=count
    return result


def validate_graph_non_gemm_scalars(payload,inventory,model,execution,ln_shared_bytes):
    """Check every known scalar argument while leaving pointers/structs unadmitted."""
    validate_graph_inventory_structure(inventory,lanes=execution['lanes'],sm=execution['device']['sm'])
    if (not isinstance(payload,dict) or type(payload.get('schema_version')) is not int or
        payload['schema_version']!=1 or
        payload.get('scope')!='captured_kernel_u32_not_pointers_or_structs_or_admission' or
        payload.get('production_admitted') is not False or
        payload.get('parameter_values_checked') is not False):
        raise ValueError('invalid scalar coverage scope')
    descriptions=decode_kernel_symbols([f['mangled_name'] for f in inventory['functions']])
    expected=expected_graph_non_gemm_scalars(model,execution,ln_shared_bytes)
    lanes=payload.get('lanes')
    if not isinstance(lanes,list) or len(lanes)!=execution['lanes']:
        raise ValueError('incomplete scalar lane coverage')
    captures={lane['lane']:lane for lane in inventory['lanes']}
    def integer(value,high):
        if type(value) is not int or not 0<=value<=high:raise ValueError('invalid scalar integer')
        return value
    seen=set();checked=0
    for lane in lanes:
        if not isinstance(lane,dict):raise ValueError('invalid scalar lane')
        lid=integer(lane.get('lane'),execution['lanes']-1)
        if lid in seen:raise ValueError('duplicate scalar lane')
        seen.add(lid);native=captures[lid]['kernels'];nodes=lane.get('kernels')
        if not isinstance(nodes,list) or len(nodes)!=len(native):
            raise ValueError('incomplete scalar node coverage')
        ids=set();actual=Counter()
        for node in nodes:
            if not isinstance(node,dict):raise ValueError('invalid scalar node')
            index=integer(node.get('kernel_index'),len(native)-1)
            if index in ids:raise ValueError('duplicate scalar node')
            ids.add(index);fid=integer(node.get('function_id'),len(descriptions)-1)
            if fid!=native[index]['function_id']:raise ValueError('scalar kernel function mismatch')
            desc=descriptions[fid];family=desc['family'];wanted=SCALAR_INDICES.get(family)
            values=node.get('u32_values')
            if not isinstance(values,list):raise ValueError('missing scalar values')
            if wanted is None:
                if family!='attention' and 'stages' not in desc:raise ValueError('unsupported scalar kernel family')
                if node.get('scalar_signature_known') is not False or values:
                    raise ValueError('opaque kernel cannot claim scalar coverage')
                continue
            if node.get('scalar_signature_known') is not True or len(values)!=len(wanted):
                raise ValueError('incomplete known scalar signature')
            pairs=[]
            for value in values:
                if not isinstance(value,dict) or set(value)!={'index','value'}:
                    raise ValueError('invalid scalar observation')
                pairs.append((integer(value['index'],32),integer(value['value'],2**32-1)))
            pairs.sort()
            if tuple(i for i,v in pairs)!=wanted:raise ValueError('scalar argument indices mismatch')
            # Same-symbol full-token and CLS kernels share the argument ABI.
            # A global multiset cannot bind their rows to the native node.
            if family in ('ln','bias_ln'):
                rows=dict(pairs)[4 if family=='ln' else 5]
                if native[index]['grid']!=[rows,1,1]:
                    raise ValueError('scalar rows differ from native launch grid')
            if family in ('fused_input','gather_cls','quantize_graph'):
                arguments=dict(pairs)
                if family=='fused_input':rows=arguments[6]*57
                elif family=='gather_cls':rows=arguments[3]
                else:rows=(arguments[5]*24+255)//256
                if native[index]['grid']!=[rows,1,1]:
                    raise ValueError('scalar batch differs from native launch grid')
            actual[(family,tuple(pairs))]+=1;checked+=1
        if actual!=expected:raise ValueError('captured scalar values differ from independent profile')
    return dict(non_gemm_u32_values_checked=True,checked_scalar_kernel_nodes=checked,
        parameter_values_checked=False,production_admitted=False)
