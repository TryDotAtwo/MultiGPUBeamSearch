# Stream1 Piece-Transformer FF1 Fused CUTLASS 2026-07-03

## Change

- Added a transformer-only FF1 fused CUTLASS path in `cuda/stream1_transformer.cu`.
- The first feed-forward projection now runs as GEMM + bias + SiLU using `GemmUniversalWithBroadcast` and `LinearCombinationBiasElementwise`.
- The existing MLP Stream1 path is unchanged.
- The residual GEMM path is unchanged.
- No distillation path was added.

## Local Docker Verification

Command summary:

```bash
cmake --build build-stream1-opt --target stream1_transformer_cuda_tests stream_benchmark -j2
./build-stream1-opt/stream1_transformer_cuda_tests
./build-stream1-opt/contract_tests
./build-stream1-opt/dispatcher_cuda_tests
BEAM_WEIGHT_DIR=test_results/stream1_transformer_reference/weights_fp16 \
BEAM_STREAM_BENCH_REPORT=test_results/stream1_transformer_ff1_fused_sm86_contended_512x1_2026-07-03.md \
BEAM_STREAM1_TRANSFORMER_B_MICRO=512 \
BEAM_STREAM1_TRANSFORMER_CONCURRENCY=1 \
./build-stream1-opt/stream_benchmark 0
```

Results:

- `stream1_transformer_cuda_tests=pass`
- `contract_tests=pass`
- `dispatcher_cuda_tests=pass`
- local SM86 smoke: `582529.0` candidates/s at `b_micro=512`, `concurrency=1`, `scratch_bytes=163864576`

Context:

- Host GPU check after the run showed RTX 3070 Laptop GPU in `P0`, `74C`, `710MiB/8192MiB`, low utilization. Earlier local low-speed readings in this turn were discarded because the GPU was occupied by a game process and sitting in `P3`.
- T4 throughput still needs Kaggle validation after pushing this branch.