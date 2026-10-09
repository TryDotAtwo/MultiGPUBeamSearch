"""Independent attention launch multiset; remaining non-GEMM and admission separate."""
from collections import Counter
import json
from tools.attention_storage_contract import validate_attention_storage_table
from tools.cuda_kernel_symbol import decode_kernel_symbols
from tools.graph_inventory_contract import validate_graph_inventory_structure


def launch_key(description,grid,block,shared):
    return (json.dumps(description,sort_keys=True,separators=(',',':')),
            tuple(grid),tuple(block),shared)


def expected_graph_attention(model,execution,storage):
    contract=dict(dtype='fp16',num_layers=4,nhead=8,head_dim=32,seq_len=57)
    if any(type(model.get(k)) is not type(v) or model[k]!=v for k,v in contract.items()):
        raise ValueError('unsupported attention model contract')
    if execution.get('executor')!='native_cuda_graph':
        raise ValueError('attention graph executor required')
    sm=execution['device']['sm']
    if type(sm) is not int or sm<75 or 75<sm<80:
        raise ValueError('unsupported attention SM')
    arch=75 if sm==75 else 80
    table=validate_attention_storage_table(storage)
    policies=execution['requested_launch_policies']
    def selected(name,default):
        value=policies.get('BEAM_STREAM1_TRANSFORMER_'+name)
        return default if value is None or value=='' else value
    def flag(name):
        value=selected(name,'0')
        if value not in ('0','1'): raise ValueError('invalid attention flag')
        return value=='1'
    tile=selected('ATTENTION_TILE_POLICY','q64k64')
    if tile not in ('q64k64','q32k64','q64k64v4'):
        raise ValueError('unsupported attention tile')
    maxk=selected('ATTENTION_MAX_K_POLICY','padded64')
    if maxk not in ('padded64','exact32'): raise ValueError('unsupported attention MaxK')
    cls=selected('CLS_ATTENTION_POLICY','cutlass')
    if cls not in ('cutlass','q32k64'): raise ValueError('unsupported CLS attention')
    outer,inner=execution['outer_microbatch'],execution['transformer_microbatch']
    seq=execution.get('expected_padded_seq_len')
    if (any(type(x) is not int or not 0<x<=65536 for x in (outer,inner))
            or inner>outer or type(seq) is not int or seq not in (57,64)):
        raise ValueError('invalid attention chunk layout')
    final=flag('FINAL_CLS_ONLY') and flag('FINAL_CLS_ATTENTION')
    result=Counter()
    for offset in range(0,outer,inner):
        batch=min(inner,outer-offset)
        for layer in range(4):
            last=final and layer==3
            q=(32 if cls=='q32k64' else 64) if last else (32 if tile=='q32k64' else 64)
            k=64 if last or maxk=='padded64' else 32
            aligned=True if last else tile!='q64k64v4'
            params=['cutlass::half_t','cutlass::arch::Sm'+str(arch),
                    'true' if aligned else 'false',str(q),'64',str(k),
                    'false','false','DefaultToBatchHook']
            desc=dict(family='attention',architecture=arch,tile=[q,64,k],parameters=params)
            record=table[(arch,q,64,k,aligned)]
            queries=1 if last else seq
            result[launch_key(desc,((queries+q-1)//q,8,batch),record['block'],
                              record['dynamic_shared_bytes'])]+=1
    return result


def validate_graph_attention_profile(inventory,model,execution,storage):
    validate_graph_inventory_structure(inventory,lanes=execution['lanes'],sm=execution['device']['sm'])
    descriptions=decode_kernel_symbols([f['mangled_name'] for f in inventory['functions']])
    expected=expected_graph_attention(model,execution,storage)
    for lane in inventory['lanes']:
        actual=Counter()
        for node in lane['kernels']:
            desc=descriptions[node['function_id']]
            if desc['family']=='attention':
                actual[launch_key(desc,node['grid'],node['block'],node['dynamic_shared_bytes'])]+=1
        if actual!=expected:
            raise ValueError('captured attention profile mismatch')
    return dict(attention_profile_checked=True,production_admitted=False,
                kernel_coverage_complete=False)
