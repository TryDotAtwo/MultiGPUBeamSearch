# Cube-444 transformer activation parity fix — 2026-08-02

## Confirmed root cause

A100 diagnostic job `33339` loaded a Cube-444 manifest with `activation=relu`. The explicit LibTorch backend consumed that field and produced first score keys around 31k–34k. Both the native CUDA backend and the Python parity oracle instead hardcoded SiLU and agreed with each other around 17k–20k, with a maximum score difference of about 14 from LibTorch. Thus LibTorch/Kaggle was the correct reference and the native/oracle agreement was a shared bug.

## Change

- Parse `piece_transformer.activation` into `Stream1ModelConfig` and propagate it through `Stream1TransformerDims`.
- Dispatch the existing fused CUTLASS `GemmUniversalWithBroadcast` FF1 epilogue to `ReLu<float>` or `SiLu<float>`; no extra materialization or pointwise kernel was added.
- Dispatch the Python parity oracle from the same manifest field.
- Reject unsupported activation names.
- Preserve p900 SiLU and leave the MLP backend unchanged.

## Local verification

- `PYTHONPATH=. python tests/test_stream1_transformer_exporter.py`: 9/9 passed.
- `python tests/test_stream1_transformer_parity.py`: 8/8 passed, including ReLU/SiLU/unknown activation dispatch.
- `git diff --check`: passed before documentation update.

## Required A100 gate

The Windows checkout has no nvcc/CUTLASS build environment. The pushed commit must therefore be built on A100 and compared against LibTorch using the real Cube-444 weights. Acceptance requires native and LibTorch score-key dumps to match under the existing parity harness; performance is measured only after correctness passes.