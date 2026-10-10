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
unaligned projections, graph gather/scatter, normalization and hop attention
remain ATen. Auto selection uses LibTorch on T4 and CUTLASS on SM80 or newer.
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
Random test weights provide no evidence of trained search quality. Large-graph
performance and multi-GPU scaling of this model family have not been measured.
