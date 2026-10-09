"""Independent non-GEMM argument offsets/sizes; no values or admission proof."""
from tools.cuda_kernel_symbol import decode_kernel_symbols
from tools.graph_inventory_contract import validate_graph_inventory_structure

PATTERNS={
    'fused_input':['pointer']*4+['network','pointer','u32','u32']+['pointer']*3,
    'ln':['pointer']*4+['u32']*2,
    'bias_ln':['pointer']*5+['u32']*2,
    'gather_cls':['pointer']*2+['dims','u32','u32'],
    'cls_ln':['pointer']*5+['dims','u32'],
    'input_graph':['pointer']*4+['network','pointer','u32','u32'],
    'padding':['pointer','dims','u32','u32'],
    'padding_legacy':['pointer','dims','u32'],
    'quantize_graph':['pointer']*5+['u32']*3+['dims','pointer']}


def validate_parameter_storage(storage):
    if not isinstance(storage,dict) or set(storage)!={'schema_version','scope','pointer','u32','dims','network','ln_shared_bytes'}:
        raise ValueError('malformed parameter storage table')
    if type(storage['schema_version']) is not int or storage['schema_version']!=1 or storage['scope']!='compiled_parameter_storage_not_admission':
        raise ValueError('parameter storage scope mismatch')
    for name in ('pointer','u32','dims','network'):
        item=storage[name]
        if not isinstance(item,dict) or set(item)!={'size','alignment'}:raise ValueError('malformed compiled type')
        size,align=item['size'],item['alignment']
        if (type(size) is not int or not 0<size<=32768 or type(align) is not int or
                align not in (1,2,4,8,16) or size%align):raise ValueError('invalid compiled type layout')
    if storage['pointer']!=dict(size=8,alignment=8) or storage['u32']!=dict(size=4,alignment=4):
        raise ValueError('unsupported pointer/integer ABI')
    if type(storage['ln_shared_bytes']) is not int or storage['ln_shared_bytes'] not in (16,24):
        raise ValueError('invalid compiled LN storage')
    return storage


def expected_non_gemm_parameters(family,storage):
    storage=validate_parameter_storage(storage)
    if family not in PATTERNS:raise ValueError('opaque or unsupported parameter ABI')
    result=[];end=0
    for index,name in enumerate(PATTERNS[family]):
        item=storage[name];align=item['alignment'];offset=(end+align-1)//align*align
        end=offset+item['size']
        if end>32768:raise ValueError('parameter layout exceeds bound')
        result.append(dict(index=index,offset=offset,size=item['size']))
    return result


def validate_non_gemm_parameter_layout(payload,inventory,execution,storage,attention_storage=None,gemm_storage=None):
    validate_parameter_storage(storage)
    validate_graph_inventory_structure(inventory,lanes=execution['lanes'],sm=execution['device']['sm'])
    if (type(payload.get('schema_version')) is not int or payload['schema_version']!=1 or
            payload.get('scope')!='captured_parameter_layout_not_values_or_admission' or
            payload.get('production_admitted') is not False or payload.get('parameter_values_checked') is not False):
        raise ValueError('parameter layout scope mismatch')
    descriptions=decode_kernel_symbols([f['mangled_name'] for f in inventory['functions']])
    attention_table=None
    if attention_storage is not None:
        from tools.attention_parameter_storage import validate_attention_parameter_storage
        attention_table=validate_attention_parameter_storage(attention_storage)
    gemm_table=None
    if gemm_storage is not None:
        from tools.gemm_parameter_storage import validate_gemm_parameter_storage,descriptor_key
        sm=execution['device']['sm']
        if type(sm) is not int:raise ValueError('invalid GEMM layout device SM')
        expected_scope=('compiled_SM75_baseline_gemm_parameters_not_admission' if sm==75 else
            'compiled_SM80_classic_gemm_parameters_not_admission' if sm in (80,86) else None)
        if expected_scope is None or type(gemm_storage) is not dict or gemm_storage.get('scope')!=expected_scope:
            raise ValueError('GEMM argument table differs from supported device route')
        gemm_table=validate_gemm_parameter_storage(gemm_storage)
    functions=payload.get('functions')
    if not isinstance(functions,list) or len(functions)!=len(descriptions):raise ValueError('parameter function count mismatch')
    checked=0;attention_checked=0;gemm_checked=0;seen=set()
    for record in functions:
        if not isinstance(record,dict) or set(record)!={'function_id','signature_known','parameters'}:
            raise ValueError('malformed parameter function')
        fid=record['function_id']
        if type(fid) is not int or not 0<=fid<len(descriptions) or fid in seen:raise ValueError('invalid parameter function ID')
        seen.add(fid)
        params=record['parameters'];end=0
        if type(record['signature_known']) is not bool or not isinstance(params,list) or len(params)>64:
            raise ValueError('malformed parameter array')
        for index,item in enumerate(params):
            if not isinstance(item,dict) or set(item)!={'index','offset','size'}:raise ValueError('malformed parameter record')
            if any(type(item[k]) is not int for k in item):raise ValueError('invalid parameter integer')
            if item['index']!=index or item['offset']<end or not 0<item['size']<=32768 or item['offset']+item['size']>32768:
                raise ValueError('invalid parameter bounds')
            end=item['offset']+item['size']
        family=descriptions[fid]['family']
        if family in PATTERNS:
            if record['signature_known'] is not True or params!=expected_non_gemm_parameters(family,storage):
                raise ValueError('captured non-GEMM parameter layout mismatch')
            checked+=1
        elif family=='attention' and attention_table is not None:
            desc=descriptions[fid]
            key=(desc['architecture'],*desc['tile'],desc['parameters'][2]=='true')
            if record['signature_known'] is not True or params!=attention_table[key]:
                raise ValueError('captured attention parameter layout mismatch')
            attention_checked+=1
        elif 'stages' in descriptions[fid] and gemm_table is not None:
            wanted=gemm_table.get(descriptor_key(descriptions[fid]))
            if wanted is None or record['signature_known'] is not True or params!=wanted:
                raise ValueError('captured GEMM parameter layout mismatch')
            gemm_checked+=1
        elif not params and record['signature_known']:raise ValueError('empty known opaque parameter layout')
    if not checked:raise ValueError('no independently checked parameter layouts')
    if attention_table is not None and not attention_checked:
        raise ValueError('no independently checked attention parameter layouts')
    if gemm_table is not None and not gemm_checked:raise ValueError('no independently checked GEMM parameter layouts')
    return dict(non_gemm_parameter_layout_checked=True,checked_functions=checked,
        attention_parameter_layout_checked=attention_checked>0,checked_attention_functions=attention_checked,
        gemm_parameter_layout_checked=gemm_checked>0,checked_gemm_functions=gemm_checked,
        all_function_argument_layouts_checked=checked+attention_checked+gemm_checked==len(descriptions),
        parameter_values_checked=False,opaque_gemm_attention_layout_checked=False,production_admitted=False)
