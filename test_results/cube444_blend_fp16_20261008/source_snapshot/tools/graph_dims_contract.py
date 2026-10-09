"""Known embedded dimension values; pointer members/full admission excluded."""
from collections import Counter
import json
from tools.cuda_kernel_symbol import decode_kernel_symbols
from tools.graph_inventory_contract import validate_graph_inventory_structure
from tools.graph_non_gemm_profile import expected_graph_non_gemms

INDICES={'fused_input':4,'input_graph':4,'gather_cls':2,'quantize_graph':8,
         'padding':1,'padding_legacy':1,'cls_ln':5}
SCOPE='captured_embedded_dims_not_pointer_members_or_admission'

def validate_graph_embedded_dims(payload,inventory,model,execution,ln_shared_bytes):
    validate_graph_inventory_structure(inventory,lanes=execution['lanes'],sm=execution['device']['sm'])
    geometry=expected_graph_non_gemms(model,execution,ln_shared_bytes)
    expected_counts=Counter()
    for (description,grid,block,shared),count in geometry.items():
        family=json.loads(description)['family']
        if family in INDICES:expected_counts[family]+=count
    expected=dict(state_len=96,num_classes=6,num_pieces=56,max_piece_size=3,seq_len=57,
        padded_seq_len=execution['expected_padded_seq_len'],
        sequence_alignment=1 if execution['expected_padded_seq_len']==57 else 16,
        d_model=256,nhead=8,head_dim=32,transformer_layers=4,ff_dim=1024,
        output_dim=24,dtype=0,activation=1)
    for key in ('state_len','num_classes','num_pieces','max_piece_size'):
        if key in model and (type(model[key]) is not int or model[key]!=expected[key]):
            raise ValueError('unsupported embedded dimension model')
    if model.get('activation','relu')!='relu':raise ValueError('unsupported embedded activation')
    if (not isinstance(payload,dict) or type(payload.get('schema_version')) is not int or
        payload['schema_version']!=1 or payload.get('scope')!=SCOPE or
        payload.get('production_admitted') is not False or payload.get('parameter_values_checked') is not False):
        raise ValueError('invalid embedded dimension scope')
    lanes=payload.get('lanes')
    if not isinstance(lanes,list) or len(lanes)!=execution['lanes']:raise ValueError('missing dimension lanes')
    captures={lane['lane']:lane for lane in inventory['lanes']}
    descriptions=decode_kernel_symbols([f['mangled_name'] for f in inventory['functions']])
    seen=set();checked=0
    for lane in lanes:
        if (not isinstance(lane,dict) or type(lane.get('lane')) is not int or
            lane['lane'] not in captures or lane['lane'] in seen):
            raise ValueError('invalid dimension lane')
        lid=lane['lane'];seen.add(lid);native=captures[lid]['kernels'];nodes=lane.get('kernels')
        if not isinstance(nodes,list) or len(nodes)!=len(native):raise ValueError('incomplete dimension nodes')
        ids=set();counts=Counter()
        for node in nodes:
            if not isinstance(node,dict) or set(node)!={'kernel_index','function_id','dims_signature_known','dims_index','dims'}:
                raise ValueError('invalid dimension node')
            index=node['kernel_index'];fid=node['function_id']
            if type(index) is not int or not 0<=index<len(native) or index in ids:
                raise ValueError('invalid dimension kernel index')
            ids.add(index)
            if type(fid) is not int or fid!=native[index]['function_id']:raise ValueError('dimension function mismatch')
            family=descriptions[fid]['family'];arg=INDICES.get(family)
            if arg is None:
                if node['dims_signature_known'] is not False or node['dims_index'] is not None or node['dims'] is not None:
                    raise ValueError('opaque dimension coverage promoted')
                continue
            dims=node['dims']
            if (node['dims_signature_known'] is not True or type(node['dims_index']) is not int or
                node['dims_index']!=arg or not isinstance(dims,dict) or set(dims)!=set(expected) or
                any(type(dims[key]) is not int or dims[key]!=value for key,value in expected.items())):
                raise ValueError('captured embedded dimensions differ from independent model')
            counts[family]+=1;checked+=1
        if counts!=expected_counts:raise ValueError('missing known embedded dimensions')
    return dict(embedded_dims_checked=True,checked_dimension_kernel_nodes=checked,
        parameter_values_checked=False,production_admitted=False)
