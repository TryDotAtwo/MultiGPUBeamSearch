from pathlib import Path


SOURCE = Path(__file__).parents[1] / "cuda" / "stream1_transformer.cu"


def test_fp16_relu_ff1_uses_fused_bias_activation_epilogue() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    dispatch_start = source.index("void stream1_transformer_ff1_linear_bias_activation_dispatch(")
    dispatch_end = source.index("constexpr std::uint32_t STREAM1_TRANSFORMER_SEQ51", dispatch_start)
    dispatch = source[dispatch_start:dispatch_end]

    assert "stream1_transformer_ff1_linear_bias_relu_cuda(" in dispatch
    assert "stream1_transformer_linear_bias_cuda(\n            input, block.ff1_weight" not in dispatch
    assert "stream1_transformer_relu_kernel<<<" not in dispatch

