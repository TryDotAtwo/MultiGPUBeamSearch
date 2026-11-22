# Two-level pancake GNN

`PancakeGNN` implements both Level-1 encoders from issue #5. Set
`encoder_type="value_graph"` for the dual-stream GATv2 encoder (default) or
`encoder_type="transformer"` for the value/position Transformer. Both use the
same multi-hop neighbor GNN and scalar value head. The runtime uses PyTorch and
the native operators shipped with this project.

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
    encoder_type="value_graph",
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
With the Transformer encoder, its self-attention and feedforward layers run through
LibTorch. The shared neighbor GNN can use the aligned CUTLASS projections.
An explicit unsupported CUTLASS device is rejected.

Scope: distinct values 0..n-1, identity target, complete pancake prefix-flip
generator set (arbitrary declared generator order is preserved). State-dict
names match the author's corresponding encoder and shared Level-2 model.
Specify the checkpoint's encoder type, dimensions and neighbor settings. Random and policy
sampling are currently rejected. Internal state batches are bounded so the
reference's stochastic frontier cap never activates; no hop is silently
removed. A cap smaller than one root's full expansion is rejected.

CPU checks cover the reference encoder, both GNN levels, TorchScript, FP16
export, blend membership and immutable snapshots. Native feature parity passed
on T4 through LibTorch and on RTX 3060 through both LibTorch and CUTLASS.
Full beam-search acceptance is recorded in `test_results/issue5_review_20261010/`.
Those October acceptance runs used random weights. The November
[Pancake-10 comparison](../../test_results/issue5_training_20261106/REPORT.md)
trains both variants and the supported Pilgrim baseline with Pilgrim's upstream
trainer and compares replay-verified solutions on its 100 published starts.
The report gives the training budget, beam, checkpoints and individual paths.


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
