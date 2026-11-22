# Large two-level pancake GNN: matched inference comparison

GPU: RTX 3060. Source: 83e46be081cae82c071128713bb93c47178f9cf3.
The model configuration matches the example in issue5's author repository:
n=100, d_model=256, two GAT layers, two neighbor hops,
max_neighbors_per_hop=75, adaptive deterministic stratified sampling,
max_frontier_states=100000. The configured neighbor counts and complete
runtime/artifact fingerprints are in result.json.

One model, same seeded synthetic weights (42), same state inputs per batch.
FP16 body, identical FP32 scalar final readout for all three implementations.
Python uses eager PyTorch F.linear; native LibTorch and native CUTLASS use
the same persistent C++ GNN loader. CUTLASS replaces aligned body projections;
other graph/tensor operations remain ATen. This benchmark does not time the
production fused blend readout epilogue or the remaining beam-search streams.
Compilation, loading, state generation and transfers are excluded. One stream,
two warmups after correctness evaluation, five synchronized timing repetitions.
Wall latency and CUDA event latency are both archived.

| State batch | Python/PyTorch | Native LibTorch | Native CUTLASS |
|---|---:|---:|---:|
| 1 | 241.94 ms | 189.47 ms | 261.61 ms |
| 2 | 468.81 ms | 368.71 ms | 468.15 ms |
| 3 | 700.55 ms | 549.54 ms | 671.27 ms |
| 4 | OOM | OOM | OOM |

LibTorch wins on this workload, including the largest measured fitting batch3.
At batch3 CUTLASS takes 1.22x the LibTorch time, Python/PyTorch 1.27x.
Peak allocated memory at batch3: Python 9.65 GB, native paths 9.35 GB;
peak reserved memory approximately 10.02 GB. Batch4 exceeds available VRAM
for a single unchunked forward; larger outer batches can still be chunked.

All fitting cases passed finite output and FP16 baseline parity. Maximum scalar
error against eager Python/PyTorch: 0.00007981 for LibTorch and 0.00023398 for
CUTLASS. This is synthetic inference acceptance, not trained search quality.
No universal speed claim or exact bottleneck attribution: no kernel/timeline
profile was captured. Use inference_backend="libtorch" for this configuration;
GPU generation alone does not select its fastest GNN backend.

The batch3 refinement reused the identical artifact and compiled native binary.
Its metadata and results are separate in result-extra.json. Reproduction scripts
and executed source are archived; no checkpoint tensors were copied to Windows.
