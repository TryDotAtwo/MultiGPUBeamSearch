from copy import deepcopy
import pytest
from tools.graph_pointer_contract import expected_pointer_roles


def test_split_qkv_query_gather_has_independent_roles_and_extents():
    _,model,execution=fixture()
    execution['device']['sm']=86
    execution['requested_launch_policies']['BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV']='1'
    roles=expected_pointer_roles(model,execution,16)
    assert roles[('gather_cls',('lane_context','lane_attention'),(58368,1024))]==1

LN='_ZN4beam44stream1_transformer_layernorm256_copy_kernelEPK6__halfPS0_S2_S2_jj'
BIAS='_ZN4beam49stream1_transformer_bias_layernorm256_copy_kernelEP6__halfS1_PKS0_S3_S3_jj'
GATHER='_ZN4beam40stream1_transformer_gather_cls256_kernelEPK6__halfPS0_NS_22Stream1TransformerDimsEjj'
QUANTIZE='_ZN4beam51stream1_transformer_score_quantize_graph_job_kernelEPK6__halfS2_PKjS4_PjjjjNS_22Stream1TransformerDimsES5_'
INPUT='_ZN4beam59stream1_transformer_build_input_layernorm256_generic_kernelILb0EEEvPKNS_11StatePackedEPKmPKjS7_NS_29Stream1TransformerNetworkViewEP6__halfjjSA_PKS9_SC_'

def fixture():
    model=dict(dtype='fp16',d_model=256,ff_dim=1024,num_layers=4,seq_len=57,
               output_dim=24,nhead=8,head_dim=32)
    execution=dict(executor='native_cuda_graph',outer_microbatch=2,transformer_microbatch=2,
        lanes=1,device={'sm':75},expected_padded_seq_len=57,
        requested_launch_policies={f'BEAM_STREAM1_TRANSFORMER_{k}':'1' for k in
            ['FUSED_INPUT_LAYERNORM','FINAL_CLS_ONLY','FINAL_CLS_ATTENTION']})
    # Independent literal roles for four blocks, final CLS attention, batch2.
    entries=[(LN,['lane_tokens','lane_context','block0_ln1_gamma','block0_ln1_beta'],[58368,58368,512,512])]
    for block in range(3):
        if block:
            entries.append((BIAS,['lane_tokens','lane_context',f'block{block-1}_ff2_bias',
                f'block{block}_ln1_gamma',f'block{block}_ln1_beta'],[58368,58368,512,512,512]))
        entries.append((BIAS,['lane_tokens','lane_context',f'block{block}_attn_out_bias',
            f'block{block}_ln2_gamma',f'block{block}_ln2_beta'],[58368,58368,512,512,512]))
    entries += [(BIAS,['lane_tokens','lane_context','block2_ff2_bias','block3_ln1_gamma','block3_ln1_beta'],[58368,58368,512,512,512]),
                (GATHER,['lane_tokens','lane_qkv'],[58368,1024]),
                (BIAS,['lane_qkv','lane_context','block3_attn_out_bias','block3_ln2_gamma','block3_ln2_beta'],[1024,1024,512,512,512]),
                (BIAS,['lane_qkv','lane_context','block3_ff2_bias','output_ln_gamma','output_ln_beta'],[1024,1024,512,512,512])]
    kernels=[dict(kernel_index=i,mangled_name=s,pointer_signature_known=True,
        pointers=[dict(index=j,role=r,offset=0,available_bytes=n) for j,(r,n) in enumerate(zip(roles,sizes))])
        for i,(s,roles,sizes) in enumerate(entries)]
    kernels.append(dict(kernel_index=len(kernels),mangled_name=INPUT,pointer_signature_known=True,
        pointers=[dict(index=i,role=r,offset=0,available_bytes=n) for i,r,n in
            [(0,'frontier_states',336),(1,'parent_base',8),(2,'active_count',4),
             (3,'graph_job_index',4),(5,'lane_tokens',58368),(8,'null',0),(9,'null',0),(10,'null',0)]]))
    kernels.append(dict(kernel_index=len(kernels),mangled_name=QUANTIZE,pointer_signature_known=True,
        pointers=[dict(index=i,role=r,offset=0,available_bytes=n)
        for i,r,n in [(0,'lane_logits',96),(1,'output_bias',48),(2,'active_count',4),
                      (3,'graph_job_index',4),(4,'score_keys',192),(9,'numeric_error',4)]]))
    payload=dict(schema_version=1,scope='captured_top_level_pointer_roles_not_extents_or_structs_or_admission',
        production_admitted=False,parameter_values_checked=False,lanes=[dict(lane=0,kernels=kernels)])
    return payload,model,execution

def check(*args):
    from tools.graph_pointer_contract import validate_graph_pointer_roles
    return validate_graph_pointer_roles(*args,ln_shared_bytes=16)

def test_independent_roles_and_minimum_extent():
    result=check(*fixture())
    assert result['known_pointer_roles_checked'] is True
    assert result['production_admitted'] is False

@pytest.mark.parametrize('mutation',['null_nonzero','nonnull_role','missing_input','short_frontier','wrong_base'])
def test_input_pointer_roles_and_required_nulls_reject_mutations(mutation):
    value,model,execution=fixture();nodes=value['lanes'][0]['kernels'];p=nodes[-2]['pointers']
    if mutation=='null_nonzero':p[5]['available_bytes']=2
    elif mutation=='nonnull_role':p[5].update(role='lane_context',available_bytes=58368)
    elif mutation=='missing_input':nodes.pop(-2);nodes[-1]['kernel_index']=len(nodes)-1
    elif mutation=='short_frontier':p[0]['available_bytes']=335
    elif mutation=='wrong_base':p[1]['role']='active_count'
    with pytest.raises(ValueError):check(value,model,execution)

def test_dual_input_requires_actual_normalized_output_and_weights():
    value,model,execution=fixture()
    execution['requested_launch_policies']['BEAM_STREAM1_TRANSFORMER_DUAL_INPUT_LN']='1'
    nodes=value['lanes'][0]['kernels'];nodes.pop(0)
    for index,node in enumerate(nodes):node['kernel_index']=index
    input_node=nodes[-2];input_node['mangled_name']=INPUT.replace('ILb0E','ILb1E')
    for pointer,role,size in zip(input_node['pointers'][5:],
        ['lane_context','block0_ln1_gamma','block0_ln1_beta'],[58368,512,512]):
        pointer.update(role=role,available_bytes=size)
    assert check(value,model,execution)['checked_pointer_kernel_nodes']==11
    input_node['pointers'][5].update(role='null',available_bytes=0)
    with pytest.raises(ValueError):check(value,model,execution)

def test_padded64_input_grid_uses_logical57_not_storage64_rows():
    from tools.graph_pointer_contract import validate_graph_pointer_roles
    value,model,execution=fixture();execution['expected_padded_seq_len']=64
    for node in value['lanes'][0]['kernels']:
        for pointer in node['pointers']:
            if pointer['available_bytes']==58368:pointer['available_bytes']=65536
    grids={0:[[row,1,1] for row in [128,128,128,128,128,128,128,2,2,2,114,1]]}
    name='stream1_transformer_zero_padded_rows_kernel'
    symbol=f'_ZN4beam{len(name)}{name}EP6__halfNS_22Stream1TransformerDimsEjj'
    nodes=value['lanes'][0]['kernels']
    for role in ['lane_tokens']*4+['lane_context']*3:
        nodes.append(dict(kernel_index=len(nodes),mangled_name=symbol,pointer_signature_known=True,
            pointers=[dict(index=0,role=role,offset=0,available_bytes=65536)]))
        grids[0].append([14,1,1])
    assert validate_graph_pointer_roles(value,model,execution,16,kernel_grids=grids)['pointer_geometry_bound'] is True
    grids[0][10][0]=128
    with pytest.raises(ValueError):
        validate_graph_pointer_roles(value,model,execution,16,kernel_grids=grids)

def test_quantize_signature_requires_all_six_pointer_roles():
    value,model,execution=fixture()
    node=dict(kernel_index=len(value['lanes'][0]['kernels']),mangled_name=QUANTIZE,
        pointer_signature_known=True,pointers=[dict(index=i,role=r,offset=0,available_bytes=n)
        for i,r,n in [(0,'lane_logits',96),(1,'output_bias',48),(2,'active_count',4),
                      (3,'graph_job_index',4),(4,'score_keys',192),(9,'numeric_error',4)]])
    value['lanes'][0]['kernels'][-1]=node
    node['kernel_index']=len(value['lanes'][0]['kernels'])-1
    assert check(value,model,execution)['known_pointer_roles_checked'] is True

@pytest.mark.parametrize('mutation',['error_index','missing_error','wrong_count','short_keys','short_logits','short_bias'])
def test_quantize_rejects_wrong_binding_or_access_extent(mutation):
    value,model,execution=fixture();pointers=value['lanes'][0]['kernels'][-1]['pointers']
    if mutation=='error_index':pointers[-1]['index']=8
    elif mutation=='missing_error':pointers.pop()
    elif mutation=='wrong_count':pointers[2]['role']='graph_job_index'
    elif mutation=='short_keys':pointers[4]['available_bytes']=191
    elif mutation=='short_logits':pointers[0]['available_bytes']=95
    elif mutation=='short_bias':pointers[1]['available_bytes']=47
    with pytest.raises(ValueError):check(value,model,execution)

def test_two_lanes_full_and_tail_chunk_coverage():
    value,model,execution=fixture()
    execution.update(outer_microbatch=5,transformer_microbatch=2,lanes=2)
    literal=value['lanes'][0]['kernels'];nodes=[]
    for _ in range(3):
        for node in deepcopy(literal):
            node['kernel_index']=len(nodes)
            if node['mangled_name']==QUANTIZE:
                node['pointers'][4]['available_bytes']=480
            if node['mangled_name']==INPUT:node['pointers'][0]['available_bytes']=672
            nodes.append(node)
    value['lanes']=[dict(lane=0,kernels=nodes),dict(lane=1,kernels=deepcopy(nodes))]
    result=check(value,model,execution)
    assert result['checked_pointer_kernel_nodes']==72
    # A full-size allocation is legitimate for the final partial chunk, but
    # it cannot substitute for a missing chunk or duplicate lane identity.
    missing=deepcopy(value);missing['lanes'][1]['kernels']=missing['lanes'][1]['kernels'][:-12]
    with pytest.raises(ValueError):check(missing,model,execution)
    duplicate=deepcopy(value);duplicate['lanes'][1]['lane']=0
    with pytest.raises(ValueError):check(duplicate,model,execution)

def test_pointer_roles_are_bound_to_actual_node_rows():
    from tools.graph_pointer_contract import validate_graph_pointer_roles
    value,model,execution=fixture()
    # Literal per-node launch rows for this independently constructed fixture.
    grids={0:[[rows,1,1] for rows in [114,114,114,114,114,114,114,2,2,2,114,1]]}
    result=validate_graph_pointer_roles(value,model,execution,16,kernel_grids=grids)
    assert result['pointer_geometry_bound'] is True
    # Both nodes have the same function symbol. Whole-node role multisets still
    # match, but CLS pointer roles cannot serve a114-row full-token kernel.
    nodes=value['lanes'][0]['kernels'];nodes[1],nodes[8]=nodes[8],nodes[1]
    nodes[1]['kernel_index']=1;nodes[8]['kernel_index']=8
    with pytest.raises(ValueError):
        validate_graph_pointer_roles(value,model,execution,16,kernel_grids=grids)

@pytest.mark.parametrize('mutation',['missing_lane','bool_lane','missing_node','bool_dim',
                                    'zero_dim','extra_axis','wrong_cls_rows','wrong_quantize_grid'])
def test_pointer_geometry_rejects_invalid_native_binding(mutation):
    from tools.graph_pointer_contract import validate_graph_pointer_roles
    value,model,execution=fixture()
    grids={0:[[row,1,1] for row in [114,114,114,114,114,114,114,2,2,2,114,1]]}
    if mutation=='missing_lane':grids={}
    elif mutation=='bool_lane':grids={False:grids[0]}
    elif mutation=='missing_node':grids[0].pop()
    elif mutation=='bool_dim':grids[0][0][0]=True
    elif mutation=='zero_dim':grids[0][0][0]=0
    elif mutation=='extra_axis':grids[0][0].append(1)
    elif mutation=='wrong_cls_rows':grids[0][8][0]=114
    elif mutation=='wrong_quantize_grid':grids[0][-1][0]=2
    with pytest.raises(ValueError):
        validate_graph_pointer_roles(value,model,execution,16,kernel_grids=grids)

def test_pointer_geometry_two_lanes_full_allocated_tail():
    from tools.graph_pointer_contract import validate_graph_pointer_roles
    value,model,execution=fixture()
    execution.update(outer_microbatch=5,transformer_microbatch=2,lanes=2)
    literal=deepcopy(value['lanes'][0]['kernels']);nodes=[]
    for _ in range(3):
        for node in deepcopy(literal):
            node['kernel_index']=len(nodes)
            if node['mangled_name']==QUANTIZE:node['pointers'][4]['available_bytes']=480
            if node['mangled_name']==INPUT:node['pointers'][0]['available_bytes']=672
            nodes.append(node)
    value['lanes']=[dict(lane=0,kernels=nodes),dict(lane=1,kernels=deepcopy(nodes))]
    rows=[114,114,114,114,114,114,114,2,2,2,114,1]*2+[57,57,57,57,57,57,57,1,1,1,57,1]
    grids={lane:[[row,1,1] for row in rows] for lane in (0,1)}
    result=validate_graph_pointer_roles(value,model,execution,16,kernel_grids=grids)
    assert result['pointer_geometry_bound'] is True
    assert result['checked_pointer_kernel_nodes']==72
    # Capacity remains full-sized, but the singleton tail launch cannot be full.
    grids[1][24][0]=114
    with pytest.raises(ValueError):
        validate_graph_pointer_roles(value,model,execution,16,kernel_grids=grids)

@pytest.mark.parametrize('mutation',['swap_norm','wrong_block','wrong_buffer','short_extent','offset',
                                    'missing_node','duplicate_node','missing_lane','bool_extent','promote'])
def test_reject_pointer_contract_mutation(mutation):
    value,model,execution=deepcopy(fixture());nodes=value['lanes'][0]['kernels']
    p=nodes[0]['pointers']
    if mutation=='swap_norm':p[2]['role'],p[3]['role']=p[3]['role'],p[2]['role']
    elif mutation=='wrong_block':p[2]['role']='block1_ln1_gamma'
    elif mutation=='wrong_buffer':p[0]['role']='lane_logits'
    elif mutation=='short_extent':p[0]['available_bytes']=58367
    elif mutation=='offset':p[0]['offset']=2
    elif mutation=='missing_node':nodes.pop()
    elif mutation=='duplicate_node':nodes.append(deepcopy(nodes[0]))
    elif mutation=='missing_lane':value['lanes']=[]
    elif mutation=='bool_extent':p[0]['available_bytes']=True
    elif mutation=='promote':value['production_admitted']=True
    with pytest.raises(ValueError):check(value,model,execution)
