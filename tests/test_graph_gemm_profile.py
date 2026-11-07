"""Independent launch arithmetic, not target-device numerical evidence."""
from copy import deepcopy
import pytest
from tools.graph_gemm_profile import expected_graph_gemms


def fixture():
    model=dict(dtype='fp16',activation='relu',d_model=256,ff_dim=1024,num_layers=4,seq_len=57,output_dim=24)
    execution=dict(executor='native_cuda_graph',device=dict(sm=86),outer_microbatch=32,
        transformer_microbatch=13,expected_padded_seq_len=57,requested_launch_policies={
            'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY':'1',
            'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION':'1',
            'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV':'1'})
    return model,execution


def test_full_and_partial_chunks_derive_counts_without_fixture_receipt():
    model,execution=fixture()
    assert sum(expected_graph_gemms(model,execution).values())==54


@pytest.mark.parametrize('outer,inner,chunks',[(1,1,1),(31,13,3),(26,13,2),(1024,1024,1)])
def test_chunk_count_varies_with_requested_batching(outer,inner,chunks):
    model,execution=fixture()
    execution.update(outer_microbatch=outer,transformer_microbatch=inner)
    assert sum(expected_graph_gemms(model,execution).values())==18*chunks


@pytest.mark.parametrize('field,value',[('dtype','fp8'),('activation','gelu'),('d_model',128),('num_layers',3)])
def test_different_model_not_silently_given_default_expectations(field,value):
    model,execution=fixture()
    model[field]=value
    with pytest.raises(ValueError): expected_graph_gemms(model,execution)


@pytest.mark.parametrize('selector,value',[('QKV_POLICY','m32n64'),('FF1_STAGES','4'),
    ('FF2_EPILOGUE','fused'),('BLOCK51','1'),('FINAL_CLS_ATTENTION','0')])
def test_unvalidated_profile_rejected(selector,value):
    model,execution=fixture()
    execution['requested_launch_policies']['BEAM_STREAM1_TRANSFORMER_'+selector]=value
    with pytest.raises(ValueError): expected_graph_gemms(model,execution)


def test_selected_ff2_tile_changes_expected_geometry():
    model,execution=fixture()
    original=expected_graph_gemms(model,execution)
    execution['requested_launch_policies']['BEAM_STREAM1_TRANSFORMER_FF2_POLICY']='m128n128'
    changed=expected_graph_gemms(model,execution)
    assert changed!=original
    assert sum(changed.values())==sum(original.values())


def test_sm75_baseline_derives_pipelined_geometry():
    model,execution=fixture();execution['device']['sm']=75
    execution['requested_launch_policies']['BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV']='0'
    result=expected_graph_gemms(model,execution)
    assert sum(result.values())==51
    assert all(key[3]==24576 for key in result)
    assert all('"stages":2' in key[0] and '"instruction":[16,8,8]' in key[0] for key in result)


def test_sm75_split_qkv_is_not_treated_as_supported():
    model,execution=fixture();execution['device']['sm']=75
    with pytest.raises(ValueError):expected_graph_gemms(model,execution)
