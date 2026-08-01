# Stream1 LibTorch Transformer Kaggle 2xT4 Benchmark v7

Date: 2026-07-04

Branch: `codex/stream1-piece-transformer`
Kernel: `trydotatwo/cayley-beam-transformer-libtorch-2xt4-benchmark`, Kaggle version 7
Model source: `vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1`

Change under test:

- Precompute per-slot positions, slot projection tensor slices, and dtype-cast slot masks once at LibTorch model load.
- Keep C++ LibTorch eager and CUDA Graph benchmark paths otherwise unchanged.

Result:

| Mode | v6 aggregate | v7 aggregate | v7 / v6 | v7 vs PyTorch ref |
|---|---:|---:|---:|---:|
| eager | 1365257.0 | 1356848.0 | 0.9938x | 1.0757x |
| cuda_graph | 1316634.0 | 1292276.0 | 0.9815x | 1.0245x |

Graph over eager aggregate in v7: `0.9524x`.

Conclusion:

The slot-cache optimization is not a speed win on Kaggle 2xT4. It was reverted in commit `93bd650`.

CUDA Graph capture/replay still works, but remains slower than eager for the working batch range. The current best verified code path remains the v6/e802183-era LibTorch eager path with pretransposed linear weights and explicit graph benchmark mode retained for further profiling.

Artifacts:

- `test_results/kaggle_libtorch_transformer_benchmark_v7_2026-07-04/stream1_libtorch_transformer_summary.json`
- `test_results/kaggle_libtorch_transformer_benchmark_v7_2026-07-04/stream1_libtorch_transformer_rows.csv`
- `test_results/kaggle_libtorch_transformer_benchmark_v7_2026-07-04/cayley-beam-transformer-libtorch-2xt4-benchmark.log`