import pytest
from tools.graph_parameter_layout import expected_non_gemm_parameters


def storage():
    return dict(schema_version=1,scope='compiled_parameter_storage_not_admission',
        pointer=dict(size=8,alignment=8),u32=dict(size=4,alignment=4),
        dims=dict(size=60,alignment=4),network=dict(size=168,alignment=8),ln_shared_bytes=16)


def test_quantizer_layout_derives_struct_alignment():
    result=expected_non_gemm_parameters('quantize_graph',storage())
    assert len(result)==10
    assert result[8]==dict(index=8,offset=52,size=60)
    assert result[9]==dict(index=9,offset=112,size=8)


def test_fused_input_carries_network_before_tail_arguments():
    result=expected_non_gemm_parameters('fused_input',storage())
    assert len(result)==11 and result[4]==dict(index=4,offset=32,size=168)


@pytest.mark.parametrize('size,alignment',[(0,4),(True,4),(60,3),(32769,4),(60,True)])
def test_malformed_compiled_storage_rejects(size,alignment):
    s=storage();s['dims']=dict(size=size,alignment=alignment)
    with pytest.raises(ValueError):expected_non_gemm_parameters('gather_cls',s)


def test_gemm_is_not_given_a_guessed_opaque_parameter_size():
    with pytest.raises(ValueError):expected_non_gemm_parameters('gemm_pipelined',storage())
