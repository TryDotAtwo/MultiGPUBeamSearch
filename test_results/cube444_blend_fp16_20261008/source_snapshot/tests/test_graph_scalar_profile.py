import pytest


def expected(**policies):
    from tools.graph_scalar_profile import expected_graph_non_gemm_scalars
    model=dict(dtype='fp16',d_model=256,ff_dim=1024,num_layers=4,seq_len=57,
               output_dim=24,nhead=8,head_dim=32)
    execution=dict(executor='native_cuda_graph',outer_microbatch=32,
        transformer_microbatch=13,expected_padded_seq_len=57,
        requested_launch_policies={'BEAM_STREAM1_TRANSFORMER_'+k:v for k,v in policies.items()})
    return expected_graph_non_gemm_scalars(model,execution,16)


def test_full_partial_chunk_scalar_values_are_not_first_row_only():
    values=expected(FUSED_INPUT_LAYERNORM='1')
    assert values[('fused_input',((6,13),(7,0)))]==1
    assert values[('fused_input',((6,13),(7,13)))]==1
    assert values[('fused_input',((6,6),(7,26)))]==1
    assert values[('ln',((4,741),(5,0)))]==2
    assert values[('ln',((4,342),(5,0)))]==1
    assert values[('bias_ln',((5,741),(6,0)))]==14
    assert values[('cls_ln',((6,13),))]==2
    assert values[('quantize_graph',((5,6),(6,32),(7,26)))]==1
    assert sum(values.values())==33


def test_final_cls_norm_uses_batch_rows_and_gather_token_stride():
    values=expected(FUSED_INPUT_LAYERNORM='1',FINAL_CLS_ONLY='1',FINAL_CLS_ATTENTION='1')
    assert values[('bias_ln',((5,741),(6,0)))]==12
    assert values[('bias_ln',((5,13),(6,0)))]==4
    assert values[('bias_ln',((5,6),(6,0)))]==2
    assert values[('gather_cls',((3,13),(4,57)))]==2
    assert values[('gather_cls',((3,6),(4,57)))]==1
    assert sum(values.values())==36


def test_unsupported_kernel_specialization_cannot_claim_scalar_coverage():
    with pytest.raises(ValueError):expected(LAYERNORM_ROWS_POLICY='persistent')
