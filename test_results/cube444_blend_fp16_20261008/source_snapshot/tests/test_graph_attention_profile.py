from collections import Counter
import pytest
from tools.graph_attention_profile import expected_graph_attention,validate_graph_attention_profile
from tests.test_attention_storage_contract import table


def execution():
    return dict(executor='native_cuda_graph',device={'sm':86},
        outer_microbatch=32,transformer_microbatch=13,expected_padded_seq_len=57,
        requested_launch_policies={})


def model():
    return dict(dtype='fp16',num_layers=4,nhead=8,head_dim=32,seq_len=57)


def test_attention_counts_include_tail_chunk():
    result=expected_graph_attention(model(),execution(),table())
    assert sum(result.values())==12
    assert sorted(result.values())==[4,8]


def test_q32_doubles_full_query_grid():
    value=execution()
    value['requested_launch_policies']['BEAM_STREAM1_TRANSFORMER_ATTENTION_TILE_POLICY']='q32k64'
    result=expected_graph_attention(model(),value,table())
    assert all(item[1][0]==2 and item[2]==(32,2,1) for item in result)


@pytest.mark.parametrize('field,value',[('head_dim',64),('nhead',0),('dtype','bf16')])
def test_unsupported_model_rejected(field,value):
    m=model();m[field]=value
    with pytest.raises(ValueError): expected_graph_attention(m,execution(),table())


def test_cls_attention_uses_one_query_and_padded_maxk():
    value=execution()
    value['requested_launch_policies'].update({
        'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY':'1',
        'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION':'1',
        'BEAM_STREAM1_TRANSFORMER_ATTENTION_TILE_POLICY':'q32k64',
        'BEAM_STREAM1_TRANSFORMER_ATTENTION_MAX_K_POLICY':'exact32'})
    result=expected_graph_attention(model(),value,table())
    assert sum(result.values())==12
    assert any(item[1][0]==1 and item[2]==(32,4,1) for item in result)


def test_empty_lane_inventory_cannot_vacuously_pass(monkeypatch):
    monkeypatch.setattr('tools.graph_attention_profile.decode_kernel_symbols',lambda symbols:[])
    value=execution();value['lanes']=2
    with pytest.raises(ValueError):
        validate_graph_attention_profile({'functions':[],'lanes':[]},model(),value,table())
