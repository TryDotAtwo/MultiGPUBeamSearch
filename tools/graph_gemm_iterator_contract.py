"""Bounded classic public C/D values; never private Params/full admission."""
import re
from tools.gemm_iterator_values import validate_epilogue_iterator,validate_mainloop_descriptor,validate_mainloop_iterator
from tools.graph_inventory_contract import validate_graph_inventory_structure

NAMES={'regular','broadcast','relu','regular75','broadcast75','relu75'}

def validate_gemm_iterator_geometry(geometry):
    if not isinstance(geometry,dict) or set(geometry)!=NAMES:
        raise ValueError('missing independent classic GEMM iterator geometry')
    seen=set()
    fields={'shape':{'row','group','cluster','tile'},'iterations':{'row','group'},
        'delta':{'row','group','cluster'},'count':{'row','group'}}
    for name,entry in geometry.items():
        if not isinstance(entry,dict) or set(entry) not in ({'symbol','descriptor'},{'symbol','descriptor','mainloop'}):
            raise ValueError('invalid independent GEMM iterator entry')
        if 'mainloop' in entry:
            if not isinstance(entry['mainloop'],dict) or set(entry['mainloop'])!={'A','B'}:
                raise ValueError('incomplete independent mainloop geometry')
            for desc in entry['mainloop'].values():validate_mainloop_descriptor(desc)
        symbol=entry['symbol'];prefix='_ZN7cutlass6KernelI' if name.startswith('regular') else '_ZN7cutlass7Kernel2I'
        if not isinstance(symbol,str) or not symbol.startswith(prefix) or len(symbol)>4096 or symbol in seen:
            raise ValueError('invalid independent GEMM iterator signature')
        seen.add(symbol);desc=entry['descriptor']
        if not isinstance(desc,dict) or set(desc)!=set(fields):
            raise ValueError('invalid independent GEMM iterator descriptor')
        for group,keys in fields.items():
            values=desc[group]
            if not isinstance(values,dict) or set(values)!=keys:
                raise ValueError('incomplete independent GEMM iterator descriptor')
            low=0 if group=='delta' else 1
            if any(type(v) is not int or not low<=v<=2**31-1 for v in values.values()):
                raise ValueError('invalid independent GEMM iterator geometry value')
    return True

def validate_graph_gemm_public_iterators(payload,inventory,execution,geometry):
    validate_gemm_iterator_geometry(geometry)
    validate_graph_inventory_structure(inventory,lanes=execution['lanes'],sm=execution['device']['sm'])
    by_symbol={entry['symbol']:entry['descriptor'] for entry in geometry.values()}
    if (not isinstance(payload,dict) or type(payload.get('schema_version')) is not int or
        payload['schema_version']!=1 or payload.get('scope')!='captured_known_gemm_members_not_complete_values_or_admission' or
        payload.get('production_admitted') is not False or payload.get('parameter_values_checked') is not False):
        raise ValueError('invalid captured GEMM iterator scope')
    lanes=payload.get('lanes');captures={lane['lane']:lane['kernels'] for lane in inventory['lanes']}
    if not isinstance(lanes,list) or len(lanes)!=execution['lanes']:
        raise ValueError('incomplete captured GEMM iterator lanes')
    seen=set();checked=0
    for lane in lanes:
        if (not isinstance(lane,dict) or type(lane.get('lane')) is not int or
            lane['lane'] not in captures or lane['lane'] in seen):
            raise ValueError('invalid captured GEMM iterator lane')
        seen.add(lane['lane']);native=captures[lane['lane']];nodes=lane.get('kernels')
        if not isinstance(nodes,list) or len(nodes)!=len(native):
            raise ValueError('incomplete captured GEMM iterator nodes')
        ids=set();lane_checked=0
        for node in nodes:
            if not isinstance(node,dict):raise ValueError('invalid GEMM iterator node')
            index=node.get('kernel_index');fid=node.get('function_id')
            if type(index) is not int or not 0<=index<len(native) or index in ids:
                raise ValueError('duplicate or invalid GEMM iterator node')
            ids.add(index)
            if type(fid) is not int or fid!=native[index]['function_id']:
                raise ValueError('GEMM iterator function mismatch')
            symbol=inventory['functions'][fid]['mangled_name']
            is_gemm=symbol.startswith(('_ZN7cutlass6KernelI','_ZN7cutlass7Kernel2I'))
            if not is_gemm:
                if (node.get('gemm_signature_known') is not False or node.get('parameter_index') is not None or
                    any(node.get(k)!={} for k in ('problem','pointers','scalars'))):
                    raise ValueError('non-GEMM iterator coverage promoted')
                continue
            if (node.get('gemm_signature_known') is not True or type(node.get('parameter_index')) is not int or
                node['parameter_index']!=0 or symbol not in by_symbol):
                raise ValueError('unsupported captured GEMM iterator signature')
            pointers=node.get('pointers');problem=node.get('problem')
            if not isinstance(pointers,dict) or not isinstance(pointers.get('B'),dict):
                raise ValueError('missing GEMM weight role')
            role=pointers['B'].get('role')
            if not isinstance(role,str):raise ValueError('invalid GEMM weight role')
            if role=='output_weight':width=24
            else:
                match=re.fullmatch(r'block([0-3])_(attn_qkv|ff1|attn_out|ff2)_weight',role or '')
                if not match:raise ValueError('unsupported GEMM weight role')
                width={'attn_qkv':768,'ff1':1024,'attn_out':256,'ff2':256}[match[2]]
            if not isinstance(problem,dict) or type(problem.get('n')) is not int or problem['n']!=width:
                raise ValueError('wrong independently expected GEMM output width')
            values=node.get('epilogue_iterators')
            if not isinstance(values,dict) or set(values)!={'C','D'}:
                raise ValueError('missing captured public GEMM iterator values')
            for operand in ('C','D'):
                validate_epilogue_iterator(values[operand],row_elements=width,descriptor=by_symbol[symbol])
                lane_checked+=1
        if not lane_checked:raise ValueError('empty captured public GEMM iterator coverage')
        checked+=lane_checked
    return dict(graph_gemm_public_iterators_checked=True,iterator_operands_checked=checked,
        production_admitted=False,parameter_values_checked=False)


def validate_gemm_mainloop_geometry(geometry):
    validate_gemm_iterator_geometry(geometry)
    if any(set(entry.get('mainloop',{}))!={'A','B'} for entry in geometry.values()):
        raise ValueError('missing independent A/B mainloop geometry')
    return True


def validate_graph_gemm_mainloop_iterators(payload,inventory,execution,geometry):
    validate_gemm_mainloop_geometry(geometry)
    validate_graph_gemm_public_iterators(payload,inventory,execution,geometry)
    by_symbol={entry['symbol']:entry['mainloop'] for entry in geometry.values()}
    checked=0
    for lane in payload['lanes']:
        for node in lane['kernels']:
            if node['gemm_signature_known'] is not True:
                if node.get('mainloop_iterators') not in (None,{}):
                    raise ValueError('non-GEMM mainloop coverage promoted')
                continue
            symbol=inventory['functions'][node['function_id']]['mangled_name']
            role=node['pointers']['B']['role']
            if role=='output_weight':k,n=256,24
            else:
                match=re.fullmatch(r'block([0-3])_(attn_qkv|ff1|attn_out|ff2)_weight',role)
                if not match:raise ValueError('unsupported mainloop weight role')
                family=match[2];k=1024 if family=='ff2' else 256
                n={'attn_qkv':768,'ff1':1024,'attn_out':256,'ff2':256}[family]
            problem=node['problem']
            if type(problem.get('k')) is not int or problem['k']!=k or problem['n']!=n:
                raise ValueError('wrong independently expected mainloop shape')
            values=node.get('mainloop_iterators')
            if not isinstance(values,dict) or set(values)!={'A','B'}:
                raise ValueError('missing captured A/B mainloop values')
            for operand,row in (('A',k),('B',n)):
                validate_mainloop_iterator(values[operand],row_elements=row,descriptor=by_symbol[symbol][operand])
                checked+=1
    return dict(graph_gemm_mainloop_iterators_checked=True,mainloop_operands_checked=checked,
        production_admitted=False,parameter_values_checked=False)
