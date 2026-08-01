# RTX 3070 current transformer profile (2026-07-22)

Fresh Nsight Systems 2025.6.3 profile after the selected four-family `m128n128` GEMM policy. CUDA Graph nodes were explicitly traced with `--cuda-graph-trace=node`; the first opaque-graph attempt was not used for kernel ranking.

Configuration: native FP16 Stream1 transformer, `b_micro=512`, concurrency 1, final CLS-only output, full final attention, production `q64k64+padded64` FMHA. Profiled point: `8.5572 ms`, exact checksum/digest retained.

## GPU time by kernel family

| Family | GPU time | Instances | Median kernel |
|---|---:|---:|---:|
| Residual CUTLASS GEMMs (attention-out + FF2) | 27.9% | 56 | 165.536 us |
| FF1 fused GEMM + bias + SiLU | 24.0% | 28 | 644.689 us |
| QKV fused GEMM + bias | 20.6% | 28 | 421.383 us |
| Bias + round + LayerNorm256 | 11.7% | 56 | 157.408 us |
| CUTLASS FMHA q64k64 | 11.0% | 28 | 224.520 us |
| Input build + LayerNorm | 2.9% | 7 | 239.928 us |
| LayerNorm256 copy | 1.7% | 7 | 139.939 us |

GEMMs now account for 72.5% of GPU kernel time. Attention and all LayerNorm families are each secondary. The next profiling target is FF1, then QKV/residual GEMM stage count and memory/occupancy behavior; launch overhead is already amortized by CUDA Graph replay.

Artifacts:

- `test_results/local3070_transformer_current_nodes_2026-07-22.nsys-rep`
- `test_results/local3070_transformer_current_nodes_2026-07-22.kernels.txt`
- `test_results/local3070_transformer_current_nodes_2026-07-22.api.txt`
