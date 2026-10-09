"""Independent known top-level pointer roles/extents, not full admission."""
from collections import Counter
from tools.graph_non_gemm_profile import expected_graph_non_gemms
from tools.cuda_kernel_symbol import decode_kernel_symbols

SCOPE='captured_top_level_pointer_roles_not_extents_or_structs_or_admission'

def expected_pointer_roles(model,execution,ln_shared_bytes):
    expected_graph_non_gemms(model,execution,ln_shared_bytes)
    policies=execution['requested_launch_policies']
    def flag(name):return policies.get('BEAM_STREAM1_TRANSFORMER_'+name,'0')=='1'
    result=Counter()
    def add(family,roles,sizes,count=1):
        result[(family,tuple(roles),tuple(sizes))]+=count
    outer=execution['outer_microbatch'];inner=execution['transformer_microbatch']
    seq=execution['expected_padded_seq_len']
    for offset in range(0,outer,inner):
        batch=min(inner,outer-offset);full=batch*seq*512;cls=batch*512
        if flag('FUSED_INPUT_LAYERNORM') or flag('DUAL_INPUT_LN'):
            # Cube4 specialization:96 logical bytes,112 physical bytes.
            if type(model.get('state_len',96)) is not int or model.get('state_len',96)!=96:
                raise ValueError('unsupported input state specialization')
            dual=flag('DUAL_INPUT_LN')
            add('fused_input',['frontier_states','parent_base','active_count','graph_job_index','lane_tokens',
                *(['lane_context','block0_ln1_gamma','block0_ln1_beta'] if dual else ['null']*3)],
                [(outer+1)*112,8,4,4,full,*([full,512,512] if dual else [0,0,0])])
        if not flag('FUSED_INPUT_LAYERNORM') and not flag('DUAL_INPUT_LN'):
            add('input_graph',['frontier_states','parent_base','active_count','graph_job_index','lane_tokens'],
                [(outer+1)*112,8,4,4,full])
            add('ln',['lane_tokens','lane_tokens','input_ln_gamma','input_ln_beta'],[full,full,512,512])
        if not flag('DUAL_INPUT_LN'):
            add('ln',['lane_tokens','lane_context','block0_ln1_gamma','block0_ln1_beta'],[full,full,512,512])
        final=flag('FINAL_CLS_ONLY')
        for block in range(3 if final else 4):
            if block:
                add('bias_ln',['lane_tokens','lane_context',f'block{block-1}_ff2_bias',
                    f'block{block}_ln1_gamma',f'block{block}_ln1_beta'],[full,full,512,512,512])
            add('bias_ln',['lane_tokens','lane_context',f'block{block}_attn_out_bias',
                f'block{block}_ln2_gamma',f'block{block}_ln2_beta'],[full,full,512,512,512])
        if final:
            add('bias_ln',['lane_tokens','lane_context','block2_ff2_bias','block3_ln1_gamma','block3_ln1_beta'],[full,full,512,512,512])
            if flag('FINAL_CLS_SPLIT_QKV'):
                add('gather_cls',['lane_context','lane_attention'],[full,cls])
            if not flag('FINAL_CLS_ATTENTION'):
                add('gather_cls',['lane_context','lane_attention'],[full,cls])
            add('gather_cls',['lane_tokens','lane_qkv'],[full,cls])
            add('bias_ln',['lane_qkv','lane_context','block3_attn_out_bias','block3_ln2_gamma','block3_ln2_beta'],[cls,cls,512,512,512])
            add('bias_ln',['lane_qkv','lane_context','block3_ff2_bias','output_ln_gamma','output_ln_beta'],[cls,cls,512,512,512])
        else:
            add('cls_ln',['lane_tokens','lane_context','block3_ff2_bias','output_ln_gamma','output_ln_beta'],
                [full,cls,512,512,512])
        if seq>57:
            family='padding_legacy' if flag('LEGACY_PADDING_ZERO') else 'padding'
            layers=3 if final else 4
            add(family,['lane_tokens'],[full],count=layers+1)
            add(family,['lane_context'],[full],count=layers+int(final and not flag('FINAL_CLS_ATTENTION')))
        add('quantize_graph',['lane_logits','output_bias','active_count','graph_job_index','score_keys','numeric_error'],
            [batch*24*2,48,4,4,outer*24*4,4])
    return result

def validate_graph_pointer_roles(payload,model,execution,ln_shared_bytes,*,kernel_grids=None):
    expected=expected_pointer_roles(model,execution,ln_shared_bytes)
    if (not isinstance(payload,dict) or type(payload.get('schema_version')) is not int or
        payload['schema_version']!=1 or payload.get('scope')!=SCOPE or
        payload.get('production_admitted') is not False or payload.get('parameter_values_checked') is not False):
        raise ValueError('invalid pointer scope')
    lanes=payload.get('lanes')
    if not isinstance(lanes,list) or len(lanes)!=execution['lanes']:raise ValueError('missing pointer lanes')
    seen=set();checked=0
    def integer(x,maximum):
        if type(x) is not int or not 0<=x<=maximum:raise ValueError('invalid pointer integer')
        return x
    if kernel_grids is not None:
        if (not isinstance(kernel_grids,dict) or
            any(type(lid) is not int for lid in kernel_grids) or
            set(kernel_grids)!=set(range(execution['lanes']))):
            raise ValueError('invalid pointer geometry lanes')
    for lane in lanes:
        if not isinstance(lane,dict):raise ValueError('invalid pointer lane')
        lid=integer(lane.get('lane'),execution['lanes']-1)
        if lid in seen:raise ValueError('duplicate pointer lane')
        seen.add(lid);nodes=lane.get('kernels')
        if not isinstance(nodes,list) or not 0<len(nodes)<=65536:raise ValueError('invalid pointer kernels')
        if kernel_grids is not None:
            grids=kernel_grids[lid]
            if not isinstance(grids,list) or len(grids)!=len(nodes):
                raise ValueError('incomplete pointer geometry coverage')
            for grid in grids:
                if (not isinstance(grid,list) or len(grid)!=3 or
                    any(type(dim) is not int or not 1<=dim<=2**31-1 for dim in grid)):
                    raise ValueError('invalid pointer launch grid')
        descriptions=decode_kernel_symbols([n['mangled_name'] for n in nodes])
        actual=Counter();ids=set()
        for node,desc in zip(nodes,descriptions):
            index=integer(node.get('kernel_index'),len(nodes)-1)
            if index in ids:raise ValueError('duplicate pointer kernel')
            ids.add(index);family=desc['family'];pointers=node.get('pointers')
            indices={'ln':(0,1,2,3),'bias_ln':(0,1,2,3,4),'gather_cls':(0,1),
                     'fused_input':(0,1,2,3,5,8,9,10),
                     'input_graph':(0,1,2,3,5),'padding':(0,),'padding_legacy':(0,),
                     'cls_ln':(0,1,2,3,4),
                     'quantize_graph':(0,1,2,3,4,9)}.get(family)
            if indices is None:
                if node.get('pointer_signature_known') is not False or pointers!=[]:
                    raise ValueError('opaque pointer signature promoted')
                continue
            if node.get('pointer_signature_known') is not True or not isinstance(pointers,list) or len(pointers)!=len(indices):
                raise ValueError('missing pointer signature')
            roles=[];sizes=[]
            for i,p in enumerate(pointers):
                if not isinstance(p,dict) or set(p)!={'index','role','offset','available_bytes'}:
                    raise ValueError('invalid pointer endpoint')
                if integer(p['index'],32)!=indices[i] or integer(p['offset'],2**64-1)!=0:
                    raise ValueError('wrong pointer index/offset')
                if not isinstance(p['role'],str):raise ValueError('invalid pointer role')
                roles.append(p['role']);sizes.append(integer(p['available_bytes'],2**64-1))
                if p['role']=='null' and p['available_bytes']!=0:
                    raise ValueError('null pointer cannot have an allocation extent')
            matches=[key for key in expected if key[0]==family and key[1]==tuple(roles) and
                     all(size>=minimum for size,minimum in zip(sizes,key[2]))]
            # Same role tuple can occur with full and partial chunks; choose
            # the largest still-needed extent to avoid accepting short tails
            # in place of full rows. Capture order is not a semantic oracle.
            matches=[k for k in matches if actual[k]<expected[k]]
            if kernel_grids is not None:
                def launch_grid(key):
                    minimum=key[2]
                    if family in ('gather_cls','cls_ln'):rows=minimum[1]//512
                    elif family=='fused_input':rows=minimum[4]//(execution['expected_padded_seq_len']*512)*57
                    elif family=='input_graph':return [minimum[4]//512,2,1]
                    elif family in ('padding','padding_legacy'):
                        seq=execution['expected_padded_seq_len']
                        rows=minimum[0]//512//seq*(seq if family=='padding_legacy' else seq-57)
                    elif family=='quantize_graph':rows=(minimum[0]//2+255)//256
                    else:rows=minimum[0]//512
                    return [rows,1,1]
                matches=[k for k in matches if launch_grid(k)==grids[index]]
            if not matches:raise ValueError('unexpected pointer roles or insufficient extent')
            key=max(matches,key=lambda k:k[2]);actual[key]+=1;checked+=1
        if actual!=expected:raise ValueError('incomplete known pointer coverage')
    return dict(known_pointer_roles_checked=True,checked_pointer_kernel_nodes=checked,
                pointer_geometry_bound=kernel_grids is not None,
                parameter_values_checked=False,production_admitted=False)
