# Two-level pancake GNN

`PancakeGNN` is our implementation of the dual-stream GATv2 encoder and
multi-hop neighbor GNN described in issue #5. Inference does not import PyG or
the author's training script. Both levels stay enabled.

```python
import torch
import cayleypy as cp
import multigpubeamsearch as beam

n = 12
identity = list(range(n))
moves = [identity[:k][::-1] + identity[k:] for k in range(2, n + 1)]
graph = cp.CayleyGraph(cp.CayleyGraphDef.create(moves), device="cuda")

# These dimensions and sampling settings MUST match the trained checkpoint.
model = beam.PancakeGNN(n, d_model=128, num_layers=2,
    neighbors=beam.NeighborConfig(max_neighbors_per_hop=11, num_hops=2))
model.load_state_dict(torch.load("checkpoint_state_dict.pt", weights_only=True))
model.eval()

result = beam.beam_search(graph, start_state=list(reversed(identity)), predictor=model,
    beam_width=1_000_000,
    native_options=beam.NativeOptions(num_gpus=2, inference_backend="auto"))
```

A plain GNN is exported as a one-member native ensemble; combine it explicitly using
`NativeEnsemble([gnn, mlp, ...], [0.6, 0.4, ...])` with graph-compatible members.
There is no limit of two members. Coefficients retain their declared order and
are not normalized automatically.

`inference_backend="libtorch"` runs tensor projections through ATen.
`"cutlass"` runs aligned FP16 GAT/MLP projections through native CUTLASS GEMM
and uses the existing ensemble readout epilogue. Two-coordinate input layers,
unaligned projections, encoder readout normalization and hop attention remain ATen. Aligned
GAT layers additionally use native fused attention/aggregation and project
three distinct edge embeddings. Channels above 1024 retain ATen aggregation. Auto selection uses LibTorch on T4 and CUTLASS on SM80 or newer.
An explicit unsupported CUTLASS device is rejected.

Scope: distinct values 0..n-1, identity target, complete pancake prefix-flip
generator set (arbitrary declared generator order is preserved). State-dict
names match the issue's value-graph model; configuration is not inferable from
weights, so specify the checkpoint's neighbor settings. Random and policy
sampling are currently rejected. Internal state batches are bounded so the
reference's stochastic frontier cap never activates; no hop is silently
removed. A cap smaller than one root's full expansion is rejected.

CPU checks cover the reference encoder, both GNN levels, TorchScript, FP16
export, blend membership and immutable snapshots. Native feature parity passed
on T4 through LibTorch and on RTX 3060 through both LibTorch and CUTLASS.
Full beam-search acceptance is recorded in `test_results/issue5_review_20261010/`.
Random test weights provide no evidence of trained search quality. Multi-GPU scaling and trained search performance of this model family have not
been measured.


Measured optimized backend: on RTX 3060, n=100, d_model=256, two layers,
two neighbor hops and state batch3, CUTLASS takes 193.153 ms versus LibTorch
540.440 ms. This is 3.377x faster than the original CUTLASS implementation.
See `test_results/gnn_opt_20261010/REPORT.md` for scope and raw evidence.
Re-export existing TorchScript artifacts to enable fused GAT; raw state-dict
parameters remain compatible. Auto selection still uses GPU architecture,
not a universal measured guarantee of the fastest backend.

Further optimization: compact CUTLASS GAT now fuses residual addition,
LayerNorm and exact erf GELU, preserving FP16 materialization boundaries and
original learned parameters/epsilon. RTX3060 matched full GNN inference
179.027 ->164.364ms (8.92% more throughput). Floating-point normalization
reduction order can differ; numerical acceptance and full beam replay pass.
Re-export older TorchScript GNN artifacts to enable the new path. See
`test_results/gnn_further_20261010/REPORT.md`. Experimental parallel softmax and
wider GEMM tiles were measured and rejected; neither is enabled by default.
