# Stream1 Transformer Graph Profile 2026-07-03

## Current committed code

- Code commit under test: `e35432dc475e3eb5b900b0470fe57f72e07f3d18` (`stream1-transformer-graph-bench-e35432d`).
- Kaggle package commit: `32562e4`, notebook pins the code checkout to that tag and sets `BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1`.
- Kaggle push was attempted twice after the package update and failed with `Maximum batch GPU session count of 2 reached`.
- Visible running Kaggle kernel found: `trydotatwo/rogii-tabfm-zero-shot-context-ensemble`.

## Local graph replay benchmark

Command shape:

```bash
BEAM_WEIGHT_DIR=test_results/kaggle_stream1_transformer_2xt4_v9_qkv_fused_2026-07-03/stream1_transformer_weights_fp16 \
BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1 \
BEAM_STREAM1_TRANSFORMER_B_MICRO=1024 \
BEAM_STREAM1_TRANSFORMER_CONCURRENCY=2 \
./build-stream1-opt/stream_benchmark 991
```

Result before profiling overhead:

```text
stream1_transformer_micro b_micro=1024 concurrency=2 rows_per_launch_group=2048 ms_per_launch_group=84.3790 parents_per_sec=24271.5 candidates_per_sec=582514.8 scratch_bytes=655458304
```

## Nsight Systems graph-node profile

Profile command used `--cuda-graph-trace=node` so CUDA graph node kernels are included in `cuda_gpu_kern_sum`.

Top GPU kernel buckets at `1024x2` under profiler overhead:

```text
22.7%  CUTLASS GEMM group, 126 launches
20.4%  CUTLASS fused-epilogue GEMM group, 56 launches
14.9%  stream1_transformer_layernorm256_copy_kernel, 126 launches
13.7%  CUTLASS fused-epilogue GEMM group, 56 launches
 7.8%  CUTLASS GemmBatched group, 448 launches
 5.3%  stream1_transformer_bias_add_kernel, 112 launches
 5.3%  stream1_transformer_softmax51_kernel, 56 launches
 4.7%  stream1_transformer_pack_v51_kernel, 56 launches
 3.8%  CUTLASS GemmBatched group, 448 launches
 1.3%  stream1_transformer_build_input_kernel, 14 launches
```

Interpretation:

- Remaining gap is mostly transformer linear GEMMs plus LayerNorm/copy and attention layout kernels.
- Build-input and score quantization are not the next bottleneck.
- Launch overhead is no longer the dominant explanation once graph replay is used.

## Rejected experiment: cuBLASLt QKV

A temporary experiment replaced only the QKV linear+bias with cuBLASLt and kept graph replay timing.

Verification:

```text
stream1_transformer_cuda_tests=pass
dispatcher_cuda_tests=pass
stream1_transformer_micro b_micro=1024 concurrency=2 rows_per_launch_group=2048 ms_per_launch_group=138.5037 parents_per_sec=14786.6 candidates_per_sec=354878.7 scratch_bytes=655458304
```

Decision:

- Reject cuBLASLt QKV for now. It is slower than the committed CUTLASS QKV graph replay path (`354878.7` vs `582514.8` candidates/s locally).
- The experimental source files and CMake changes were removed; no cuBLASLt runtime path is committed.