# Stream1 Transformer CUTLASS vs LibTorch/cuBLAS Profiling

Date: 2026-07-04
Branch: codex/stream1-piece-transformer

Question: can the LibTorch/cuBLAS projection path be replaced with CUTLASS, and what does profiling show?

## Environment

- Docker image: `cmz-native-dev:2026-05-26`.
- GPU visible from Docker: NVIDIA GeForce RTX 3070 Laptop, SM86, driver 572.70.
- CUTLASS root: `/opt/cutlass`.
- Stream1 weights: `test_results/kaggle_libtorch_transformer_benchmark_v1_2026-07-04/stream1_transformer_weights_fp16`.
- Nsight Systems available in the image.
- Nsight Compute 2026.1.1 was not compatible with the host driver for metric collection; the compatible profiler was `/opt/nvidia/nsight-compute/2025.1.1/ncu` with explicit metrics.

## LibTorch/cuBLAS baseline

Local LibTorch benchmark, batch 384, 50 timed iterations:

| Mode | candidates/s | Notes |
|---|---:|---|
| eager | 578741 | `test_results/stream1_transformer_libtorch_local_nsys_profile_2026-07-04.md` |
| cuda_graph | 598639 | graph replay confirmed with `cudaGraphLaunch` |

Nsight Systems showed the LibTorch hot path is cuBLASLt GEMMs, PyTorch FlashAttention, LayerNorm, and elementwise kernels. NCU 2025.1.1 on the dominant cuBLASLt GEMM showed tensor-core HMMA use, with SM throughput around 44-46% and DRAM throughput around 49-54% for sampled launches.

Interpretation: the cuBLASLt kernels are not scalar fallback kernels; the larger cost is the LibTorch/ATen kernel mix and unfused block structure.

## Native CUTLASS path

The repository already has a native Stream1 piece-transformer CUTLASS path in `cuda/stream1_transformer.cu`:

- QKV projection uses CUTLASS `GemmUniversalWithBroadcast` with fused bias.
- FF1 projection uses CUTLASS `GemmUniversalWithBroadcast` with fused bias + SiLU.
- Residual projections use CUTLASS GEMM with residual beta.
- Attention uses CUTLASS FMHA when available.

Local native CUTLASS measurements:

| Config | candidates/s | Evidence |
|---|---:|---|
| b_micro=384, concurrency=1, eager | 871235.6 | `native_cutlass_transformer_b384_eager_2026-07-04.md` |
| b_micro=384, concurrency=1, graph bench | 867818.4 | `native_cutlass_transformer_b384_graph_2026-07-04.md` |
| b_micro=512, concurrency=2, eager sweep best | 991974.7 | `native_cutlass_transformer_sweep_2026-07-04.md` |
| b_micro=384, concurrency=1, block51 opt-in | 819299.1 | `native_cutlass_transformer_b384_block51_2026-07-04.md` |

At the same local scale, native CUTLASS is already much faster than LibTorch/cuBLASLt:

- `871235.6 / 578741 = 1.505x` vs LibTorch eager at b384.
- `991974.7 / 578741 = 1.714x` using the best local native sweep point.

## Native Nsight Systems hot path

Profile: `test_results/nsight_native_cutlass_transformer_2026-07-04/native_b384_eager.log`.

GPU kernel shares at b384/c1:

| Kernel family | GPU time share |
|---|---:|
| CUTLASS plain GEMM (`Kernel`) | 31.1% |
| CUTLASS fused epilogue GEMM (`Kernel2`) | 26.3% + 19.4% |
| LayerNorm/copy kernels | 10.3% + 2.5% |
| CUTLASS attention | 8.3% |
| build input | 1.9% |
| score quantize | ~0.0% |

Interpretation: replacing LibTorch cuBLAS with CUTLASS is not just possible; the native path already does it and is faster locally. The remaining bottleneck is the native transformer block itself: GEMM epilogue/layout/memory behavior plus LayerNorm, not launch overhead.

## Native Nsight Compute counters

NCU 2025.1.1 profiles:

- `test_results/nsight_native_cutlass_transformer_2026-07-04/ncu2025_native_Kernel2_b512_c2.ncu-rep`
- `test_results/nsight_native_cutlass_transformer_2026-07-04/ncu2025_native_Kernel_b512_c2.ncu-rep`

Summarized counters:

| Kernel class | sampled launches | SM throughput avg | DRAM throughput avg | HMMA / tensor cycles |
|---|---:|---:|---:|---:|
| Kernel2 fused epilogue | 6 | 41.24% | 66.17% | 1.0000 |
| Kernel plain GEMM | 3 | 40.28% | 70.35% | 1.0000 |
| Kernel2 in mixed capture | 3 | 40.63% | 64.82% | 1.0000 |

Interpretation: native CUTLASS GEMMs use tensor cores correctly. They are not compute-saturating; DRAM/epilogue/layout pressure is high. Tuning should target epilogue/layout and block-level fusion, not a simple library swap.

## Tried: wider CUTLASS N tile

Experiment: changed transformer QKV and FF1 fused GEMMs from threadblock/warp shapes `128x64x32 / 64x32x32` to `128x128x32 / 64x64x32`.

Result at b_micro=512, concurrency=2:

| Shape | candidates/s |
|---|---:|
| baseline 128x64x32 | 991974.7 |
| experimental 128x128x32 | 957325.9 |

The experiment regressed and was reverted. Evidence: `native_cutlass_transformer_b512_c2_tile128n_2026-07-04.md`.

## Current conclusion

- A direct `at::linear`/cuBLAS -> CUTLASS question has a concrete answer: yes, but not inside LibTorch. The clean route is the existing native CUTLASS Stream1 transformer backend.
- On local RTX 3070, native CUTLASS is already 1.5x-1.7x faster than the local LibTorch/cuBLAS path.
- The next speed work should continue on native CUTLASS production path and tune the dominant GEMM/epilogue/LayerNorm structure.
- The current 128x64 CUTLASS tile is better than the tried 128x128 tile on this shape.
- CUDA Graph is not the primary win for native transformer at this point; graph b384 was slightly slower than eager b384.

Next candidates:

1. Profile/tune separate QKV, FF1, attention-out, and FF2 GEMM shapes instead of one shared tile policy.
2. Try transformer-specific epilogues that combine residual, bias, and next LayerNorm without extra global-memory round trips, but only if correctness stays exact enough against the reference score-key contract.
3. Run the same native CUTLASS sweep on 2xT4 Kaggle/A100 because local RTX 3070 behavior differs from Kaggle v19/v6 results.