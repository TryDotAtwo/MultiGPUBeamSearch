# Own two-level pancake GNN acceptance

Implemented both inference backends without changing the five beam-search streams.
Source tested on RTX 3060: `ff6c74e6e72aa08391d9e2bc1b8db60a7a67cb01`.
The native feature gate used `3b706c92458528802ab3ab33f8344c7b9ed1adff`;
the later commit only changes Python artifact/ensemble wrapping, not native kernels.

| Check | Result |
|---|---|
| CPU reference encoder, n=4/6/12 | exact |
| CPU complete two-level network | max error 1.49e-8 |
| Focused artifact, device, cache and build tests | 37 passed |
| Backend regressions | 52 passed, 2 platform skips |
| Native LibTorch features, T4 on Kaggle | max error 0.00048828125 |
| Native LibTorch features, RTX 3060 | max error 0.0003662109375 |
| Native CUTLASS features, RTX 3060 | max error 0.0008544921875 |
| Full beam: single GNN, LibTorch / CUTLASS | both found replay-valid path length 3 |
| Full beam: three GNNs, LibTorch / CUTLASS | both found replay-valid path length 3 |
| Three-head readout against FP32 projection oracle | max score-key error 0 on both backends |

Full-beam fixture: n=4, complete prefix flips, start [3,1,0,2], beam=1024,
max_steps=16, one RTX 3060, d_model=32, layers=2, neighbor hops=2,
max_frontier_states=24, seeds 11/12/13, blend coefficients 0.2/0.3/0.5.
The small expansion cap forces native feature chunking; both GNN levels stay enabled.
`replay_valid` verifies the returned moves against the input graph and target.
Both native runners and the ensemble calibration executable compiled successfully.

Kaggle gate: https://www.kaggle.com/code/trydotatwo/gnn-native-backend-gate-20261010
T4 checks LibTorch only; CUTLASS execution there is explicitly NOT_RUN.

Scope and limits:
- Own inference implementation; no PyG runtime or execution of the author's training script.
- Complete pancake graph with distinct values 0..n-1 and identity target.
- Checkpoint configuration must be supplied; stochastic/policy neighbor sampling is rejected.
- FP16 projections and parameters, FP32 softmax/aggregation where specified.
- CUTLASS handles aligned dense GNN projections and the blend readout epilogue;
  gathers, reductions, normalization, attention and small unaligned projections remain ATen.
- Synthetic weights establish implementation correctness, not trained search quality.
- Large-graph performance, automatic-profile GPU acceptance and multi-GPU GNN scaling
  were not measured here. The batch-4 tiny fixture's CUTLASS latency exceeded LibTorch;
  those numbers are not evidence of a throughput benefit for production GNN sizes.
- The existing broader model suite could not collect locally because the installed
  CayleyPy lacks ModelConfig; do not interpret the focused counts as all-repository coverage.

Raw remote results, build/native logs and their SHA256 manifest are archived alongside
this report. No checkpoint tensors were downloaded to Windows.
