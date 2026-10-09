"""Known consumed NetworkView pointers, not opaque Params or admission."""
from collections import Counter
import json
from tools.cuda_kernel_symbol import decode_kernel_symbols
from tools.graph_inventory_contract import validate_graph_inventory_structure
from tools.graph_non_gemm_profile import expected_graph_non_gemms
from tools.network_member_consumption import CONSUMED

SCOPE = 'captured_consumed_network_members_not_opaque_params_or_admission'


def validate_graph_network_members(payload, inventory, model, execution, ln_shared_bytes):
    validate_graph_inventory_structure(inventory, lanes=execution['lanes'],
                                      sm=execution['device']['sm'])
    geometry = expected_graph_non_gemms(model, execution, ln_shared_bytes)
    # Literal Cube4 specialization; reject other models rather than extrapolate ABI.
    for name, value in [('state_len', 96), ('num_classes', 6),
                        ('num_pieces', 56), ('max_piece_size', 3)]:
        if type(model.get(name, value)) is not int or model.get(name, value) != value:
            raise ValueError('unsupported consumed network model')
    expected_counts = Counter()
    for (description, grid, block, shared), count in geometry.items():
        family = json.loads(description)['family']
        if family in ('fused_input', 'input_graph'):
            expected_counts[family] += count
    required = dict(fast_slot_projected=3*6*256*2,
                    fast_piece_static=56*256*2, cls_token=256*2,
                    piece_positions=56*3*2, piece_mask=56*3)
    fused_required = dict(required, input_ln_gamma=256*2, input_ln_beta=256*2)
    if set(required)!=CONSUMED['input_graph'] or set(fused_required)!=CONSUMED['fused_input']:
        raise ValueError('consumed NetworkView reference/source audit disagreement')
    if (not isinstance(payload, dict) or type(payload.get('schema_version')) is not int or
            payload['schema_version'] != 1 or payload.get('scope') != SCOPE or
            payload.get('production_admitted') is not False or
            payload.get('parameter_values_checked') is not False):
        raise ValueError('invalid consumed network scope')
    lanes = payload.get('lanes')
    if not isinstance(lanes, list) or len(lanes) != execution['lanes']:
        raise ValueError('missing consumed network lanes')
    captures = {lane['lane']: lane for lane in inventory['lanes']}
    descriptions = decode_kernel_symbols([f['mangled_name'] for f in inventory['functions']])
    seen = set()
    for lane in lanes:
        if (not isinstance(lane, dict) or type(lane.get('lane')) is not int or
                lane['lane'] not in captures or lane['lane'] in seen):
            raise ValueError('invalid consumed network lane')
        lid = lane['lane']; seen.add(lid)
        native = captures[lid]['kernels']; nodes = lane.get('kernels')
        if not isinstance(nodes, list) or len(nodes) != len(native):
            raise ValueError('incomplete consumed network nodes')
        ids = set(); counts = Counter()
        for node in nodes:
            if not isinstance(node, dict) or set(node) != {
                    'kernel_index', 'function_id', 'network_signature_known',
                    'network_index', 'members'}:
                raise ValueError('invalid consumed network node')
            index = node['kernel_index']; fid = node['function_id']
            if type(index) is not int or not 0 <= index < len(native) or index in ids:
                raise ValueError('invalid consumed network kernel index')
            ids.add(index)
            if type(fid) is not int or fid != native[index]['function_id']:
                raise ValueError('consumed network function mismatch')
            family = descriptions[fid]['family']
            if family not in ('fused_input', 'input_graph'):
                if (node['network_signature_known'] is not False or
                        node['network_index'] is not None or node['members'] != []):
                    raise ValueError('opaque network coverage promoted')
                continue
            if (node['network_signature_known'] is not True or
                    type(node['network_index']) is not int or node['network_index'] != 4):
                raise ValueError('wrong captured NetworkView argument')
            wanted = fused_required if family == 'fused_input' else required
            members = node['members']
            if not isinstance(members, list) or len(members) != len(wanted):
                raise ValueError('missing consumed network members')
            names = set()
            for member in members:
                if not isinstance(member, dict) or set(member) != {
                        'member', 'role', 'offset', 'available_bytes'}:
                    raise ValueError('invalid consumed network endpoint')
                name = member['member']
                if not isinstance(name, str) or name not in wanted or name in names:
                    raise ValueError('unexpected or duplicate consumed member')
                names.add(name)
                if (member['role'] != name or type(member['offset']) is not int or
                        member['offset'] != 0 or type(member['available_bytes']) is not int or
                        not wanted[name] <= member['available_bytes'] <= 2**64-1):
                    raise ValueError('wrong consumed member role or extent')
            counts[family] += 1
        if counts != expected_counts:
            raise ValueError('incomplete known consumed network coverage')
    return dict(consumed_network_members_checked=True, production_admitted=False,
                parameter_values_checked=False)
