# Stream1 LibTorch Transformer Kaggle 2xT4 Benchmark v3

Date: 2026-07-04

Branch: `codex/stream1-piece-transformer`
Commit: `9481ce2`
Kernel: `trydotatwo/cayley-beam-transformer-libtorch-2xt4-benchmark`, Kaggle version 3
Model source: `vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1`

Change under test:

- Restored dense gather+mask token build after v2 active-index regression.
- Kept `c10::InferenceMode` in the C++ benchmark process.
- Kept broader batch sweep `[128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048]`.

Result:

| GPU | Best batch | Candidates/s |
|---:|---:|---:|
| 0 | 320 | 484660.0 |
| 1 | 384 | 486948.0 |
| aggregate | | 971608.0 |

Comparison:

| Baseline | Aggregate candidates/s | Ratio |
|---|---:|---:|
| LibTorch v1 dense token build | 947512.0 | 1.0254x |
| LibTorch v2 active-index token build | 917025.0 | 1.0595x |
| PyTorch batch_process reference | 1261394.0 | 0.7703x |
| native/CUTLASS v19 | 1224735.7 | 0.7933x |

Conclusion:

`c10::InferenceMode` plus the broader batch sweep gives a small but real improvement over v1, while the active-index token path remains rejected. The next likely C++/Python mismatch is the explicit Q/K/V `.contiguous()` calls before LibTorch SDPA, which may add three copies per transformer block.

Artifacts:

- `test_results/kaggle_libtorch_transformer_benchmark_v3_2026-07-04/stream1_libtorch_transformer_summary.json`
- `test_results/kaggle_libtorch_transformer_benchmark_v3_2026-07-04/stream1_libtorch_transformer_rows.csv`
- `test_results/kaggle_libtorch_transformer_benchmark_v3_2026-07-04/cayley-beam-transformer-libtorch-2xt4-benchmark.log`