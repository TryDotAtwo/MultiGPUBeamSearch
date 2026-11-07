"""Independent bounded transfer argument checks; never production admission.

Node enumeration is unordered. This validates the exact transfer multiset,
not CUDA dependency ordering or the authenticity of a captured sidecar.
"""
from collections import Counter
import json


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid {name}")
    return value


def validate_bound_score_graph_transfers(value, expected, content_hashes):
    """Bind transfer observations to independently read artifact contents."""
    if not isinstance(value, dict) or expected.get('executor') != 'native_cuda_graph':
        raise ValueError('captured graph transfers require graph execution')
    def frozen(item):
        return json.dumps(item, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if 'device' not in expected or frozen(value.get('device')) != frozen(expected['device']):
        raise ValueError('graph transfers device mismatch')
    for field, filename in [('execution_sha256', 'execution.json'),
                            ('inventory_sha256', 'graph_kernel_inventory.json')]:
        wanted = content_hashes.get(filename)
        if not isinstance(wanted, str) or len(wanted) != 64 or value.get(field) != wanted:
            raise ValueError('graph transfers content binding mismatch: '+field)
    return validate_score_graph_transfers(value, outer=expected['outer_microbatch'],
        inner=expected['transformer_microbatch'], lanes=expected['lanes'])


def validate_score_graph_transfers(value, *, outer, inner, lanes):
    """Require every lane's two zero resets and all 24-output FP16 chunks."""
    outer = _integer(outer, 1, 2**32 - 1, "outer")
    inner = _integer(inner, 1, outer, "inner")
    lanes = _integer(lanes, 1, 64, "lanes")
    # Keep hostile metadata validation bounded independently of GPU capacity.
    chunks = (outer + inner - 1) // inner
    if chunks > 100000:
        raise ValueError("transfer inventory exceeds supported node bound")
    if not isinstance(value, dict):
        raise ValueError("transfer inventory must be an object")
    if (type(value.get("schema_version")) is not int or
            value["schema_version"] != 1 or
            value.get("scope") != "captured_graph_transfers_not_admission" or
            value.get("production_admitted") is not False):
        raise ValueError("invalid transfer inventory contract")
    entries = value.get("lanes")
    if not isinstance(entries, list) or len(entries) != lanes:
        raise ValueError("incomplete lane inventory")
    expected = Counter({("memset", "raw_scores", 0, outer * 48, 0): 1,
                        ("memset", "numeric_error", 0, 4, 0): 1})
    for start in range(0, outer, inner):
        expected[("memcpy", "lane_logits", 0, "raw_scores", start * 48,
                  min(inner, outer - start) * 48, "device_to_device")] += 1
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid lane")
        lane = _integer(entry.get("lane"), 0, lanes - 1, "lane")
        if lane in seen:
            raise ValueError("duplicate lane")
        seen.add(lane)
        operations = entry.get("operations")
        if not isinstance(operations, list) or len(operations) != chunks + 2:
            raise ValueError("incomplete transfer multiset")
        actual = Counter()
        for op in operations:
            if not isinstance(op, dict):
                raise ValueError("invalid operation")
            kind = op.get("kind")
            size = _integer(op.get("bytes"), 1, outer * 48, "bytes")
            offset = _integer(op.get("destination_offset"), 0, outer * 48,
                              "destination offset")
            destination = op.get("destination")
            if not isinstance(destination, str) or destination not in ("raw_scores", "numeric_error"):
                raise ValueError("unsupported destination")
            if kind == "memset":
                element = _integer(op.get("element_size"), 1, 4, "element size")
                if element not in (1, 2, 4) or size % element:
                    raise ValueError("invalid normalized memset span")
                reset = _integer(op.get("value"), 0, 0, "reset value")
                actual[(kind, destination, offset, size, reset)] += 1
            elif kind == "memcpy":
                source_offset = _integer(op.get("source_offset"), 0, inner * 48,
                                         "source offset")
                if (op.get("source") != "lane_logits" or
                        op.get("direction") != "device_to_device"):
                    raise ValueError("unsupported transfer endpoint or direction")
                actual[(kind, "lane_logits", source_offset, destination,
                        offset, size, "device_to_device")] += 1
            else:
                raise ValueError("unknown transfer operation")
        if actual != expected:
            raise ValueError("transfer arguments differ from expected chunk coverage")
    return dict(transfer_arguments_checked=True,
                transfer_pitched_geometry_checked=False,
                checked_memcpy_nodes=lanes * chunks,
                checked_memset_nodes=lanes * 2,
                dependency_order_checked=False,
                production_admitted=False)


def validate_score_graph_dependencies(value, *, outer, inner, lanes):
    """Require a unique execution order for the supported single-stream capture.

    This verifies serialized reuse, not internal kernel pointer/scalar values.
    Parallel/multi-stream specializations require a different verified contract.
    """
    result=validate_score_graph_transfers(value,outer=outer,inner=inner,lanes=lanes)
    checked=0
    for lane in value['lanes']:
        count=_integer(lane.get('node_count'),1,65536,'node count')
        operations=lane['operations'];kernel_ids=lane.get('kernel_node_ids')
        edges=lane.get('dependency_edges')
        if not isinstance(kernel_ids,list) or len(kernel_ids)>count:
            raise ValueError('missing kernel node inventory')
        if not isinstance(edges,list) or not count-1<=len(edges)<=262144:
            raise ValueError('incomplete or oversized dependency edges')
        covered=set();op_by_id={}
        for op in operations:
            node=_integer(op.get('node_id'),0,count-1,'operation node')
            if node in covered:raise ValueError('duplicate operation node')
            covered.add(node);op_by_id[node]=op
        for raw in kernel_ids:
            node=_integer(raw,0,count-1,'kernel node')
            if node in covered:raise ValueError('duplicate kernel node')
            covered.add(node)
        if len(covered)!=count:raise ValueError('node inventory not exhaustive')
        incoming=[0]*count;outgoing=[[] for _ in range(count)];seen=set()
        for edge in edges:
            if not isinstance(edge,list) or len(edge)!=2:raise ValueError('invalid edge')
            src=_integer(edge[0],0,count-1,'edge source')
            dst=_integer(edge[1],0,count-1,'edge destination')
            if src==dst or (src,dst) in seen:raise ValueError('self/duplicate edge')
            seen.add((src,dst));incoming[dst]+=1;outgoing[src].append(dst)
        ready=[node for node,degree in enumerate(incoming) if degree==0]
        order=[]
        while ready:
            if len(ready)!=1:raise ValueError('graph does not serialize scratch reuse')
            node=ready.pop();order.append(node)
            for dst in outgoing[node]:
                incoming[dst]-=1
                if incoming[dst]==0:ready.append(dst)
        if len(order)!=count:raise ValueError('cyclic/incomplete graph dependencies')
        first=[op_by_id.get(node,{}).get('kind') for node in order[:2]]
        if first!=['memset','memset']:raise ValueError('resets must precede all inference')
        copy_offsets=[];kernels_since_copy=0
        for node in order[2:]:
            if node not in op_by_id:kernels_since_copy+=1;continue
            op=op_by_id[node]
            if op['kind']!='memcpy':raise ValueError('late reset')
            if kernel_ids and not kernels_since_copy:
                raise ValueError('chunk copy must follow its inference work')
            copy_offsets.append(op['destination_offset']);kernels_since_copy=0
        if copy_offsets!=list(range(0,outer*48,inner*48)):
            raise ValueError('copy order differs from chunk order')
        if kernels_since_copy:raise ValueError('uncopied trailing inference work')
        checked+=count
    result.update(dependency_order_checked=True,checked_dependency_nodes=checked)
    return result
