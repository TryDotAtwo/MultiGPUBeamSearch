from copy import deepcopy
import pytest

SYMBOLS=[
 '_ZN4beam59stream1_transformer_build_input_layernorm256_generic_kernelILb0EEEvPKNS_11StatePackedEPKmPKjS7_NS_29Stream1TransformerNetworkViewEP6__halfjjSA_PKS9_SC_',
 '_ZN4beam44stream1_transformer_layernorm256_copy_kernelEPK6__halfPS0_S2_S2_jj',
 '_ZN4beam49stream1_transformer_bias_layernorm256_copy_kernelEP6__halfS1_PKS0_S3_S3_jj',
 '_ZN4beam40stream1_transformer_gather_cls256_kernelEPK6__halfPS0_NS_22Stream1TransformerDimsEjj',
 '_ZN4beam51stream1_transformer_score_quantize_graph_job_kernelEPK6__halfS2_PKjS4_PjjjjNS_22Stream1TransformerDimsES5_']


def fixture():
    # Hand-derived final-CLS-attention classic FP16:12 known kernels per chunk.
    model=dict(dtype='fp16',d_model=256,ff_dim=1024,num_layers=4,seq_len=57,
               output_dim=24,nhead=8,head_dim=32)
    execution=dict(executor='native_cuda_graph',outer_microbatch=32,
        transformer_microbatch=13,lanes=1,device={'sm':75},expected_padded_seq_len=57,
        requested_launch_policies={f'BEAM_STREAM1_TRANSFORMER_{k}':'1' for k in
            ['FUSED_INPUT_LAYERNORM','FINAL_CLS_ONLY','FINAL_CLS_ATTENTION']})
    nodes=[];observations=[]
    for batch,offset in [(13,0),(13,13),(6,26)]:
        entries=[(0,[(6,batch),(7,offset)]),(1,[(4,batch*57),(5,0)])]
        entries += [(2,[(5,batch*57),(6,0)])]*6
        entries += [(2,[(5,batch),(6,0)])]*2
        entries += [(3,[(3,batch),(4,57)]),(4,[(5,batch),(6,32),(7,offset)])]
        rows=[batch*57]*8+[batch,batch,batch,(batch*24+255)//256]
        for (fid,values),row in zip(entries,rows):
            observations.append(dict(kernel_index=len(nodes),function_id=fid,
                scalar_signature_known=True,u32_values=[dict(index=i,value=v) for i,v in values]))
            nodes.append(dict(function_id=fid,grid=[row,1,1],block=[128,1,1],dynamic_shared_bytes=0))
    attrs=dict(binary_version=75,ptx_version=75,max_threads_per_block=1024,num_regs=12,
        static_shared_bytes=0,local_bytes=0,constant_bytes=0,max_dynamic_shared_bytes=49152)
    inv=dict(functions=[dict(mangled_name=s,attributes=deepcopy(attrs)) for s in SYMBOLS],
        lanes=[dict(lane=0,node_count=36,node_types=dict(kernel=36,memcpy=0,memset=0),kernels=nodes)])
    value=dict(schema_version=1,scope='captured_kernel_u32_not_pointers_or_structs_or_admission',
        production_admitted=False,parameter_values_checked=False,
        lanes=[dict(lane=0,kernels=observations)])
    return value,inv,model,execution


def check(value,inv,model,execution):
    from tools.graph_scalar_profile import validate_graph_non_gemm_scalars
    return validate_graph_non_gemm_scalars(value,inv,model,execution,16)


def test_every_full_partial_scalar_checked_without_promotion():
    result=check(*fixture())
    assert result['non_gemm_u32_values_checked'] is True
    assert result['checked_scalar_kernel_nodes']==36
    assert result['parameter_values_checked'] is False
    assert result['production_admitted'] is False

def test_scalar_rows_cannot_swap_between_same_symbol_native_nodes():
    value,inv,model,execution=fixture()
    assert check(value,inv,model,execution)['non_gemm_u32_values_checked'] is True
    nodes=value['lanes'][0]['kernels']
    # Keep function IDs, argument-index schema and global scalar multiset.
    nodes[2]['u32_values'],nodes[8]['u32_values']=nodes[8]['u32_values'],nodes[2]['u32_values']
    with pytest.raises(ValueError):check(value,inv,model,execution)

@pytest.mark.parametrize('first,tail',[(0,24),(10,34),(11,35)])
def test_chunk_scalar_batch_cannot_swap_full_and_tail_nodes(first,tail):
    value,inv,model,execution=fixture()
    assert check(value,inv,model,execution)['non_gemm_u32_values_checked'] is True
    nodes=value['lanes'][0]['kernels']
    nodes[first]['u32_values'],nodes[tail]['u32_values']=nodes[tail]['u32_values'],nodes[first]['u32_values']
    with pytest.raises(ValueError):check(value,inv,model,execution)


@pytest.mark.parametrize('mutation', ['tail_batch','tail_offset','slot_batch','dtype',
    'gather_stride','cls_rows','missing_kernel','duplicate_kernel','function_id',
    'unknown_signature','index_bool','value_bool','missing_value','extra_value',
    'missing_lane','duplicate_lane','lane_bool','schema_bool','scope','promotion'])
def test_mutated_actual_scalar_contract_rejected(mutation):
    value,inv,model,execution=fixture();nodes=value['lanes'][0]['kernels']
    if mutation=='tail_batch':nodes[-1]['u32_values'][0]['value']=13
    elif mutation=='tail_offset':nodes[-1]['u32_values'][2]['value']=13
    elif mutation=='slot_batch':nodes[-1]['u32_values'][1]['value']=13
    elif mutation=='dtype':nodes[1]['u32_values'][1]['value']=1
    elif mutation=='gather_stride':nodes[-2]['u32_values'][1]['value']=64
    elif mutation=='cls_rows':nodes[-3]['u32_values'][0]['value']=342
    elif mutation=='missing_kernel':nodes.pop()
    elif mutation=='duplicate_kernel':nodes[-1]=deepcopy(nodes[0])
    elif mutation=='function_id':nodes[-1]['function_id']=0
    elif mutation=='unknown_signature':nodes[-1]['scalar_signature_known']=False
    elif mutation=='index_bool':nodes[-1]['u32_values'][0]['index']=True
    elif mutation=='value_bool':nodes[1]['u32_values'][1]['value']=False
    elif mutation=='missing_value':nodes[-1]['u32_values'].pop()
    elif mutation=='extra_value':nodes[-1]['u32_values'].append(dict(index=8,value=0))
    elif mutation=='missing_lane':value['lanes']=[]
    elif mutation=='duplicate_lane':value['lanes'].append(deepcopy(value['lanes'][0]))
    elif mutation=='lane_bool':value['lanes'][0]['lane']=False
    elif mutation=='schema_bool':value['schema_version']=True
    elif mutation=='scope':value['scope']='admitted'
    elif mutation=='promotion':value['parameter_values_checked']=True
    with pytest.raises(ValueError):check(value,inv,model,execution)
