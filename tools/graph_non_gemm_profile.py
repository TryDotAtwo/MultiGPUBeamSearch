"""Independent classic-FP16 non-GEMM launch geometry; not full ABI admission."""
from collections import Counter
from tools.graph_attention_profile import launch_key
from tools.cuda_kernel_symbol import decode_kernel_symbols
from tools.graph_inventory_contract import validate_graph_inventory_structure

FAMILIES={'fused_input','input_graph','ln','bias_ln','gather_cls','cls_ln',
          'padding','padding_legacy','quantize_graph'}


def expected_graph_non_gemms(model,execution,ln_shared_bytes):
    required=dict(dtype='fp16',d_model=256,ff_dim=1024,num_layers=4,
        seq_len=57,output_dim=24,nhead=8,head_dim=32)
    if any(type(model.get(k)) is not type(v) or model[k]!=v for k,v in required.items()):
        raise ValueError('unsupported non-GEMM model')
    if execution.get('executor')!='native_cuda_graph':raise ValueError('graph executor required')
    if type(ln_shared_bytes) is not int or ln_shared_bytes not in (16,24):
        raise ValueError('missing independently compiled LN storage')
    policies=execution['requested_launch_policies']
    def selected(name,default):
        value=policies.get('BEAM_STREAM1_TRANSFORMER_'+name)
        return default if value in (None,'') else value
    def flag(name):
        value=selected(name,'0')
        if value not in ('0','1'):raise ValueError('invalid non-GEMM flag')
        return value=='1'
    if selected('LAYERNORM_ROWS_POLICY','row')!='row' or flag('STAGE_PROFILE'):
        raise ValueError('non-GEMM specialization not independently validated')
    for name in ('ATTN_OUT_EPILOGUE','FF2_EPILOGUE'):
        if selected(name,'separate')!='separate':raise ValueError('fused epilogue not validated')
    seq=execution.get('expected_padded_seq_len')
    outer,inner=execution['outer_microbatch'],execution['transformer_microbatch']
    if type(seq) is not int or seq not in (57,64):raise ValueError('invalid padding expectation')
    if any(type(x) is not int or not 0<x<=65536 for x in (outer,inner)) or inner>outer:
        raise ValueError('invalid non-GEMM chunk profile')
    dual=flag('DUAL_INPUT_LN')
    if dual and seq!=57:raise ValueError('dual input requires compact tokens')
    final=flag('FINAL_CLS_ONLY')
    cls_attention=final and flag('FINAL_CLS_ATTENTION')
    if flag('FINAL_CLS_SPLIT_QKV') and not cls_attention:raise ValueError('split QKV dependency')
    legacy=flag('LEGACY_PADDING_ZERO')
    result=Counter()
    def add(family,grid,block=(128,1,1),shared=0,count=1,**extra):
        result[launch_key(dict(family=family,**extra),grid,block,shared)]+=count
    for offset in range(0,outer,inner):
        batch=min(inner,outer-offset);rows=batch*seq
        if dual or flag('FUSED_INPUT_LAYERNORM'):
            add('fused_input',(batch*57,1,1),shared=ln_shared_bytes,dual=dual)
        else:
            add('input_graph',(rows,2,1))
            add('ln',(rows,1,1),shared=ln_shared_bytes)
        if not dual:add('ln',(rows,1,1),shared=ln_shared_bytes)
        # Full-token LN2 once/layer plus previous-block LN1 except layer0.
        # Final CLS layer keeps token LN1 but its LN2/output LN use batch rows.
        add('bias_ln',(rows,1,1),shared=ln_shared_bytes,count=6 if final else 7)
        if final:
            add('bias_ln',(batch,1,1),shared=ln_shared_bytes,count=2)
            add('gather_cls',(batch,1,1),count=(1+int(flag('FINAL_CLS_SPLIT_QKV'))) if cls_attention else 2)
        else:
            add('cls_ln',(batch,1,1),block=(256,1,1),shared=1024)
        if seq>57:
            elements=batch*(seq if legacy else seq-57)*256
            add('padding_legacy' if legacy else 'padding',((elements+255)//256,1,1),
                block=(256,1,1),count=1+2*(3 if final else 4)+(1 if final and not cls_attention else 0))
        add('quantize_graph',((batch*24+255)//256,1,1),block=(256,1,1))
    return result


def validate_graph_non_gemm_profile(inventory,model,execution,ln_shared_bytes):
    validate_graph_inventory_structure(inventory,lanes=execution['lanes'],sm=execution['device']['sm'])
    descriptions=decode_kernel_symbols([f['mangled_name'] for f in inventory['functions']])
    expected=expected_graph_non_gemms(model,execution,ln_shared_bytes)
    for lane in inventory['lanes']:
        actual=Counter()
        for node in lane['kernels']:
            desc=descriptions[node['function_id']]
            if desc['family'] in FAMILIES:
                actual[launch_key(desc,node['grid'],node['block'],node['dynamic_shared_bytes'])]+=1
            elif desc['family']!='attention' and 'stages' not in desc:
                raise ValueError('uncovered non-GEMM function')
        if actual!=expected:raise ValueError('captured non-GEMM profile mismatch')
    return dict(non_gemm_profile_checked=True,parameter_abi_checked=False,
        kernel_coverage_complete=False,production_admitted=False)
