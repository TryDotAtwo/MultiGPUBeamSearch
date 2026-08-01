# Stream1 Transformer Graph Benchmark Mode 2026-07-03

## Change

- Added `BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1` to `tools/stream_benchmark_transformer.cu`.
- In graph mode the benchmark captures one `stream1_transformer_inference_cuda` graph per stream, instantiates it, and times repeated `cudaGraphLaunch` calls.
- This is benchmark-only instrumentation. It does not change `cuda/stream1.cu`, the MLP path, dispatcher semantics, or production solver kernels.

## Why

The non-graph microbenchmark times the host enqueue path between CUDA events. That is useful for raw launch overhead, but the solver/dispatcher uses CUDA graph replay for repeated Stream1 work. Graph timing is the closer performance gate for comparing production Stream1 transformer changes.

## Verification

Environment:

```text
Docker image: gpu-dev-cutlass-nsight:cuda128-sm120
Workspace: D:\100XH100\.worktrees\stream1-piece-transformer
Weights: test_results/kaggle_stream1_transformer_2xt4_v9_qkv_fused_2026-07-03/stream1_transformer_weights_fp16
```

Commands:

```bash
cmake --build build-stream1-opt --target stream1_transformer_cuda_tests dispatcher_cuda_tests stream_benchmark -j2
./build-stream1-opt/stream1_transformer_cuda_tests
./build-stream1-opt/dispatcher_cuda_tests
cmake --build build-stream1-opt --target contract_tests -j2
./build-stream1-opt/contract_tests
BEAM_WEIGHT_DIR=test_results/kaggle_stream1_transformer_2xt4_v9_qkv_fused_2026-07-03/stream1_transformer_weights_fp16 \
BEAM_STREAM_BENCH_REPORT=test_results/stream1_transformer_graphbench_verify_1024x2_2026-07-03.md \
BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1 \
BEAM_STREAM1_TRANSFORMER_B_MICRO=1024 \
BEAM_STREAM1_TRANSFORMER_CONCURRENCY=2 \
./build-stream1-opt/stream_benchmark 991
```

Results:

```text
stream1_transformer_cuda_tests=pass
dispatcher_cuda_tests=pass
contract_tests=pass
stream1_transformer_micro b_micro=1024 concurrency=2 rows_per_launch_group=2048 ms_per_launch_group=84.3790 parents_per_sec=24271.5 candidates_per_sec=582514.8 scratch_bytes=655458304
```

## Notes

- Earlier local experiments with SM75 FMHA, cuBLASLt QKV/FF1, and QKV tile changes were not kept because they were slower or not yet proven under graph replay.
- Local RTX measurements are still not a final T4/A100 gate. The next required check is Kaggle 2xT4 graph replay using the same benchmark mode.