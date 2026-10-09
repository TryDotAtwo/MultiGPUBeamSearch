"""Independent non-GEMM geometry, not parameter-ABI or admission proof."""
import pytest
from tools.graph_non_gemm_profile import expected_graph_non_gemms


def profile(**policies):
    return dict(executor='native_cuda_graph',outer_microbatch=32,
        transformer_microbatch=13,expected_padded_seq_len=57,
        requested_launch_policies={'BEAM_STREAM1_TRANSFORMER_'+k:v for k,v in policies.items()})


def model():
    return dict(dtype='fp16',d_model=256,ff_dim=1024,num_layers=4,seq_len=57,
        output_dim=24,nhead=8,head_dim=32)


def test_full_token_profile_counts_each_chunk():
    result=expected_graph_non_gemms(model(),profile(FUSED_INPUT_LAYERNORM='1'),16)
    assert sum(result.values())==33 # input + LN1 + 7 bias-LNs + CLS-LN + quantize


def test_final_cls_attention_has_separate_batch_norms_and_gather():
    result=expected_graph_non_gemms(model(),profile(FUSED_INPUT_LAYERNORM='1',
        FINAL_CLS_ONLY='1',FINAL_CLS_ATTENTION='1'),16)
    assert sum(result.values())==36
    assert sum(n for k,n in result.items() if 'gather_cls' in k[0])==3


def test_dual_input_removes_first_layer_ln():
    normal=expected_graph_non_gemms(model(),profile(FUSED_INPUT_LAYERNORM='1'),16)
    dual=expected_graph_non_gemms(model(),profile(DUAL_INPUT_LN='1'),16)
    assert sum(normal.values())-sum(dual.values())==3


def test_split_qkv_includes_query_input_gather_per_chunk():
    result=expected_graph_non_gemms(model(),profile(FUSED_INPUT_LAYERNORM='1',
        FINAL_CLS_ONLY='1',FINAL_CLS_ATTENTION='1',FINAL_CLS_SPLIT_QKV='1'),16)
    assert sum(n for k,n in result.items() if 'gather_cls' in k[0])==6


def test_padding_launches_cover_input_and_two_per_full_layer():
    e=profile(FUSED_INPUT_LAYERNORM='1');e['expected_padded_seq_len']=64
    result=expected_graph_non_gemms(model(),e,16)
    assert sum(n for k,n in result.items() if 'padding' in k[0])==27


@pytest.mark.parametrize('policy,value',[('LAYERNORM_ROWS_POLICY','persistent'),
    ('ATTN_OUT_EPILOGUE','fused'),('FF2_EPILOGUE','fused'),('STAGE_PROFILE','1')])
def test_unimplemented_specializations_reject(policy,value):
    with pytest.raises(ValueError):expected_graph_non_gemms(model(),profile(**{policy:value}),16)


@pytest.mark.parametrize('shared',[0,True,20,32])
def test_unrecognized_compiled_ln_storage_reject(shared):
    with pytest.raises(ValueError):expected_graph_non_gemms(model(),profile(),shared)
