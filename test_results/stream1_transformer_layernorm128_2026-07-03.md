# Stream1 Piece Transformer LayerNorm128 Optimization 2026-07-03

## Scope

Optimize only the transformer `cols==256` LayerNorm copy kernel in `cuda/stream1_transformer.cu`. MLP Stream1 files and the MLP dispatch path are not changed.

## Change

- `stream1_transformer_layernorm256_copy_kernel` now launches `128` threads per row instead of `256`.
- Each thread processes two columns: `tid` and `tid + 128`.
- Shared scratch drops from `8` float warp slots to `4` float warp slots.

## Verification

Docker build targets completed for SM75 and SM86:

```text
cmake --build build-stream1-transformer-fmha-sm75 --target stream1_transformer_cuda_tests stream_benchmark -j2
cmake --build build-stream1-transformer-fmha-sm86 --target stream1_transformer_cuda_tests dispatcher_cuda_tests contract_tests stream_benchmark -j2
```

CUDA correctness tests passed:

```text
stream1_transformer_cuda_tests=pass
dispatcher_cuda_tests=pass
contract_tests=pass
```

## Local Profiling

Nsight Systems profile after the change:

- Report: `test_results/nsys_stream1_transformer_ln128_1024x1_2026-07-03.nsys-rep`
- SQLite: `test_results/nsys_stream1_transformer_ln128_1024x1_2026-07-03.sqlite`

Top kernel split at `B_MICRO=1024`, `concurrency=1` under profiler:

```text
27.9% CUTLASS GEMM group
25.2% CUTLASS fused-epilogue GEMM group
17.4% CUTLASS fused-epilogue GEMM group
11.3% stream1_transformer_layernorm256_copy_kernel
 9.2% CUTLASS FMHA attention_kernel_batched_impl
 7.0% stream1_transformer_bias_add_kernel
 1.9% stream1_transformer_build_input_kernel
```

Previous strided-FMHA profile had LayerNorm at about `22.9%`, so this removes roughly half of the LayerNorm kernel share in the profiled graph.

## Local Benchmark Notes

Local RTX 3070 Laptop measurements were noisy because the GPU was in a hot/P5 display state. Observed standalone `B_MICRO=512`, `concurrency=2` runs included:

```text
569866.0 candidates/s
611182.1 candidates/s
346219.6 candidates/s
```

Because of that variance, final performance comparison must use Kaggle 2xT4, matching the previous v10-v12 benchmark workflow.

## Rejected Experiment

Tried fusing `attn_out` and `ff2` residual+bias through CUTLASS `GemmUniversalWithBroadcast` epilogues with `beta=1`. Correctness tests passed, but local `512x2` throughput dropped to `418414.9` candidates/s, so the code was reverted. The separate `bias_add_kernel` remains in the committed source.
