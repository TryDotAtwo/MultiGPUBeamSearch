"""Captured attention values/coverage; geometry and opaque GEMMs remain separate."""
from collections import Counter
from tools.attention_member_values import validate_attention_params
from tools.attention_optional_values import validate_attention_optional_members
from tools.cuda_kernel_symbol import decode_kernel_symbols
from tools.graph_inventory_contract import validate_graph_inventory_structure


def validate_graph_attention_members(payload,inventory,model,execution):
    validate_graph_inventory_structure(inventory,lanes=execution['lanes'],sm=execution['device']['sm'])
    for key,value in dict(dtype='fp16',num_layers=4,nhead=8,head_dim=32,seq_len=57,d_model=256).items():
        if type(model.get(key)) is not type(value) or model[key]!=value:
            raise ValueError('unsupported attention member model')
    if (not isinstance(payload,dict) or type(payload.get('schema_version')) is not int or
        payload['schema_version']!=1 or payload.get('scope')!='captured_attention_params_not_gemm_or_admission' or
        payload.get('production_admitted') is not False or payload.get('parameter_values_checked') is not False):
        raise ValueError('invalid attention member scope')
    policies=execution['requested_launch_policies']
    def flag(name):return policies.get('BEAM_STREAM1_TRANSFORMER_'+name)=='1'
    final=flag('FINAL_CLS_ONLY') and flag('FINAL_CLS_ATTENTION')
    split=flag('FINAL_CLS_SPLIT_QKV')
    expected=Counter()
    outer=execution['outer_microbatch'];inner=execution['transformer_microbatch']
    if type(outer) is not int or type(inner) is not int or not 0<inner<=outer<=65536:
        raise ValueError('invalid attention batching')
    for offset in range(0,outer,inner):
        batch=min(inner,outer-offset)
        expected[(batch,False)]+=3 if final else 4
        if final:expected[(batch,True)]+=1
    descriptions=decode_kernel_symbols([f['mangled_name'] for f in inventory['functions']])
    captures={lane['lane']:lane['kernels'] for lane in inventory['lanes']}
    lanes=payload.get('lanes')
    if not isinstance(lanes,list) or len(lanes)!=execution['lanes']:
        raise ValueError('missing attention member lanes')
    seen=set()
    for lane in lanes:
        if not isinstance(lane,dict) or type(lane.get('lane')) is not int or lane['lane'] not in captures or lane['lane'] in seen:
            raise ValueError('invalid attention member lane')
        lid=lane['lane'];seen.add(lid);nodes=lane.get('kernels');native=captures[lid]
        if not isinstance(nodes,list) or len(nodes)!=len(native):raise ValueError('incomplete attention member nodes')
        counts=Counter();ids=set()
        for node in nodes:
            if not isinstance(node,dict) or set(node)!={'kernel_index','function_id','attention_signature_known','parameter_index','pointers','scalars','optional_pointers','optional_scalars','pytorch_rng_state_present'}:
                raise ValueError('invalid attention member node')
            index=node['kernel_index'];fid=node['function_id']
            if type(index) is not int or not 0<=index<len(native) or index in ids:
                raise ValueError('invalid attention kernel index')
            ids.add(index)
            if type(fid) is not int or fid!=native[index]['function_id']:raise ValueError('attention function mismatch')
            if descriptions[fid]['family']!='attention':
                if node['optional_pointers']!={} or node['optional_scalars']!={} or node['pytorch_rng_state_present'] is not False:
                    raise ValueError('opaque optional member promotion')
                if node['attention_signature_known'] is not False or node['parameter_index'] is not None or node['pointers']!={} or node['scalars']!={}:
                    raise ValueError('opaque member promotion')
                continue
            if node['attention_signature_known'] is not True or type(node['parameter_index']) is not int or node['parameter_index']!=0:
                raise ValueError('missing typed attention argument')
            if not isinstance(node['scalars'],dict):raise ValueError('invalid attention scalars')
            validate_attention_optional_members(node)
            cls=node['scalars'].get('num_queries')==1
            batch=native[index]['grid'][2]
            validate_attention_params(node['pointers'],node['scalars'],batch=batch,
                seq=execution['expected_padded_seq_len'],cls=cls,split=split and cls)
            counts[(batch,cls)]+=1
        if counts!=expected:raise ValueError('incomplete attention member coverage')
    return True
