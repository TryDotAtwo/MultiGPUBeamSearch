# Native GNN CUTLASS optimization, RTX 3060

Matched large-model inference: n=100, d_model=256, two GAT layers,
two neighbor hops, adaptive deterministic sampling, state batch=3.
FP16 body and identical FP32 scalar readout. Synthetic seed-42 weights;
all 83 parameter tensors are bitwise identical between implementations.
Five synchronized repetitions after warmup; compilation/loading/transfers excluded.

| Implementation | Median forward latency |
|---|---:|
| Original native CUTLASS, same host | 652.268 ms |
| Optimized native CUTLASS | 193.153 ms |
| Native LibTorch, same comparison | 540.440 ms |
| Python/PyTorch | 690.569 ms |

CUTLASS improves 3.377x against its original implementation and runs 2.798x
faster than LibTorch here. This measures one complete GNN inference for three
states, not one full beam-search depth. Peak allocated memory falls to
1,760,023,040 bytes, versus 9,377,376,256 bytes for LibTorch.

Profiling identified expensive ATen gathers, index_add and intermediate
edge-message operations. Dense projections remain CUTLASS. A fused native
GAT kernel combines per-head attention, softmax, message aggregation and
head reduction; it projects only three distinct edge embeddings and avoids
materializing repeated edge features. Original edge order is restored before
FP16 accumulation. Both encoder graphs and both GNN levels remain enabled.
Beam streams, frontier semantics and Stream 4 deduplication are unchanged.

Correctness: scalar-output maximum absolute error against FP16 Python is
0.000171445. All 28 fused-kernel cases pass across n=4/100, both graph types,
and channel widths 4,8,16,64,128,256,1024. Maximum direct kernel error is
0.000732422; projection-plus-kernel error is 0.000976563.
Compute Sanitizer memcheck reports zero errors. Full production beam search
passes for a single GNN and a three-GNN blend: both find length-3 paths and
adapter replay verifies them. Focused CPU tests: 8 passed; independent schema
registration also passes a projection-only-extension compatibility check.

GPU-tested candidate: 67fbc205814854bda4b98d1d6020b8dce81c18ab.
Original baseline: 96c76779043e26348f0a99f275d3866a82f33552.
Final Python schema-registration compatibility fix does not change CUDA math.
Raw results and UTF-8 SHA256 manifest are under evidence/. Executed benchmark
and beam job are included; checkpoints remain remote, no tensors copied to Windows.

Old exported TorchScript artifacts must be re-exported to use the new fused
path. State-dict names and trained parameters remain compatible. Channel
widths above 1024 retain the ordinary aggregation path. T4 retains LibTorch.
No claim of universal 2x acceleration, trained solution quality, or multi-GPU
scaling follows from this single-RTX3060 synthetic benchmark.
