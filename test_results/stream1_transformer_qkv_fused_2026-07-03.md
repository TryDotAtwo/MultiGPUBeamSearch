# Stream1 Piece-Transformer QKV Bias Fused CUTLASS 2026-07-03

## Change

- Added a transformer-only QKV `linear+bias` CUTLASS path in `cuda/stream1_transformer.cu`.
- The attention QKV projection now writes pre-biased QKV through `GemmUniversalWithBroadcast` and `LinearCombinationBiasElementwise` with identity activation.
- Removed the old separate QKV bias kernel from the transformer path.
- Updated the SM80 BF16 FMHA pack path to consume pre-biased QKV, so it does not add QKV bias a second time.
- The MLP Stream1 path remains unchanged.
- No fallback or distillation path was added.

## Local Docker Verification

Command summary:

```bash
cmake --build build-stream1-opt --target stream1_transformer_cuda_tests stream_benchmark -j2
./build-stream1-opt/stream1_transformer_cuda_tests
./build-stream1-opt/contract_tests
./build-stream1-opt/dispatcher_cuda_tests
BEAM_WEIGHT_DIR=test_results/stream1_transformer_reference/weights_fp16 \
BEAM_STREAM_BENCH_REPORT=test_results/stream1_transformer_qkv_fused_local_smoke_2026-07-03.md \
BEAM_STREAM1_TRANSFORMER_B_MICRO=512 \
BEAM_STREAM1_TRANSFORMER_CONCURRENCY=1 \
./build-stream1-opt/stream_benchmark 0
```

Results:

- `stream1_transformer_cuda_tests=pass`
- `contract_tests=pass`
- `dispatcher_cuda_tests=pass`
- local SM86 smoke: `617951.1` candidates/s at `b_micro=512`, `concurrency=1`, `scratch_bytes=163864576`

## Context

- Previous FF1-only local smoke in `test_results/stream1_transformer_ff1_fused_2026-07-03.md`: `582529.0` candidates/s at the same `512x1` point.
- This is a local smoke, not a final T4 gate. Kaggle 2xT4 validation is still required for the actual T4 comparison.