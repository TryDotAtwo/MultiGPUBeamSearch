# GNN projection bias fused into CUTLASS, RTX 3060

The separate add_bias kernel is removed. Bias is broadcast through GEMM C
with row stride zero. The custom epilogue explicitly preserves the original
FP32 accumulator -> FP16 -> add FP16 bias -> FP16 rounding; split_k remains 1.
GEMM tile policy, GAT aggregation and beam-search architecture are unchanged.

Baseline: 28a1ba8c482926b23549cbd066b9b9829b37522e.
GPU-tested candidate: 713d7bb2cde3b0e60ad22dd9aa389b5924215528.
Same RTX3060, CUDA12.8, Torch2.11+cu128, SM86 binary hashes in evidence.
Large synthetic GNN: n100/d256, two layers, two neighbor hops (75/38),
state batch3, FP16 body and common FP32 scalar head. Seed42 weights, seed901
state inputs. Loading, compilation and transfers excluded.

Initial five-sample comparison was noisy (219.734 vs 192.138 ms). Follow-up
AB/BA order, five warmups and 21 CUDA-event repetitions per block:

| Block | Separate bias | Fused bias |
|---|---:|---:|
| A then B | 218.772 ms | 178.577 ms |
| B then A | 211.891 ms | 180.435 ms |
| Median of all 42 samples per variant | 213.548 ms | 178.821 ms |

Throughput improves 1.1942x (19.42%); latency drops 16.26%. This is one
complete GNN forward for three states, not a complete beam-search depth.
Output SHA256 is identical in every native block and in the initial runs.

28 direct projection cases bitwise match the old CUDA implementation:
rows1/3/127/128/129/4096/8192 with aligned K/N8..512; absent, zero, random,
and cancellation bias. Compute Sanitizer memcheck: zero errors. Against
Python FP16, initial scalar error is 9.17e-5 (old) and 8.26e-5 (new); the Python
graph scatter reduction is not bitwise deterministic across these runs.
The native old/new outputs are bitwise identical. No trained-quality claim.

Standalone projection timings do not establish universal acceleration:
4096x128x512: 1.542 vs 1.620 ms; 100000x128x512: 2.451 vs 2.303 ms;
100000x256x128: 1.912 vs 1.922 ms. These include allocation and launch through
the Torch operator and have noticeable host/GPU variability. The full-model
AB/BA result is the acceptance performance evidence, not these microbenchmarks.

Raw JSON, executed drivers, compiler log, binary fingerprints and SHA256
manifest are archived under evidence/. Tensor checkpoints/reference tensors
stay on the remote host until teardown; none copied to Windows. Sanitizer
profiling reports a CUPTI subscriber conflict; its memory check completes
successfully, but sanitizer-run profiler events are not performance evidence.
