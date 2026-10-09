"""Independent canonical launch values and problem multiplicities, not admission."""
from collections import Counter
import re
from tools.graph_gemm_iterator_contract import validate_graph_gemm_mainloop_iterators
from tools.gemm_launch_values import validate_gemm_launch_values,validate_epilogue_override_endpoints
from tools.gemm_head_values import validate_head_gemm
from tools.gemm_bias_values import validate_bias_gemm
from tools.gemm_residual_values import validate_residual_gemm
from tools.gemm_iterator_values import validate_epilogue_iterator


def validate_graph_gemm_launch_values(payload, inventory, execution, geometry, padded_seq_len):
    if type(padded_seq_len) is not int or padded_seq_len not in (57, 64):
        raise ValueError('unsupported independent padded sequence')
    validate_graph_gemm_mainloop_iterators(payload, inventory, execution, geometry)
    outer, inner = execution['outer_microbatch'], execution['transformer_microbatch']
    if any(type(v) is not int or not 0 < v <= 65536 for v in (outer, inner)) or inner > outer:
        raise ValueError('unsupported independent GEMM batch')
    policies = execution['requested_launch_policies']
    final = policies.get('BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY') or '0'
    if final not in ('0', '1') or (policies.get('BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV') or '0') != '0':
        raise ValueError('unsupported canonical launch route')
    expected = Counter()
    for offset in range(0, outer, inner):
        batch = min(inner, outer - offset)
        for layer in range(4):
            rows = batch if final == '1' and layer == 3 else batch * padded_seq_len
            for family, m, n, k in (
                ('attn_qkv', batch * padded_seq_len, 768, 256),
                ('attn_out', rows, 256, 256), ('ff1', rows, 1024, 256),
                ('ff2', rows, 256, 1024)):
                expected[(f'block{layer}_{family}_weight', m, n, k)] += 1
        expected[('output_weight', batch, 24, 256)] += 1
    names = {entry['symbol']: name for name, entry in geometry.items()}
    checked = 0
    for lane in payload['lanes']:
        actual = Counter()
        for node in lane['kernels']:
            if node['gemm_signature_known'] is not True:
                if node.get('launch_scalars') != {}:
                    raise ValueError('non-GEMM launch values promoted')
                continue
            symbol = inventory['functions'][node['function_id']]['mangled_name']
            name = names[symbol]
            role = node['pointers']['B']['role']
            match = re.fullmatch(r'block([0-3])_(attn_qkv|ff1|attn_out|ff2)_weight', role)
            broadcast = bool(match and match[2] in ('attn_qkv', 'ff1'))
            wanted_name = ('relu' if match and match[2] == 'ff1' else 'broadcast') if broadcast else 'regular'
            # The output head uses the fixed SM75 pipelined kernel on SM80/86 too.
            if name != wanted_name + ('75' if role == 'output_weight' or execution['device']['sm'] == 75 else ''):
                raise ValueError('GEMM family/signature mismatch')
            problem = node['problem']
            if set(problem) != {'m', 'n', 'k'} or any(type(v) is not int for v in problem.values()):
                raise ValueError('invalid GEMM problem shape')
            key = (role, problem['m'], problem['n'], problem['k'])
            if key not in expected:
                raise ValueError('unexpected independently bound GEMM shape')
            overrides = node.get('epilogue_pointers')
            validate_epilogue_override_endpoints(overrides,broadcast=broadcast)
            if role == 'output_weight':
                validate_head_gemm(node, batch=key[1])
            elif broadcast:
                validate_bias_gemm(node, layer=int(match[1]), rows=key[1],
                    kind='qkv' if match[2] == 'attn_qkv' else 'ff1')
            else:
                validate_residual_gemm(node, layer=int(match[1]), rows=key[1],
                    kind=match[2], cls=final == '1' and match[1] == '3')
            if broadcast:
                # Canonical source passes ldt=output_cols despite ptr_Tensor=null.
                # Independent probe statically binds Tensor's FP16 map to C/D's map.
                validate_epilogue_iterator(node.get('tensor_iterator'), row_elements=key[2],
                    descriptor=geometry[name]['descriptor'])
            elif node.get('tensor_iterator') != {}:
                raise ValueError('regular GEMM tensor iterator promoted')
            validate_gemm_launch_values(node, rows=key[1], columns=key[2],
                reduction=key[3], tile=(128, 64, 32), broadcast=broadcast)
            actual[key] += 1
            checked += 1
        if actual != expected:
            raise ValueError('incomplete canonical GEMM launch multiplicities')
    return dict(graph_gemm_launch_values_checked=True, launch_nodes_checked=checked,
                parameter_values_checked=False, production_admitted=False)
