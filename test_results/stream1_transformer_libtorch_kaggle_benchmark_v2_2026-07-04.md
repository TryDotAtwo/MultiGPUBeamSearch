# Stream1 LibTorch Transformer Kaggle 2xT4 Benchmark v2

Date: 2026-07-04

Branch: `codex/stream1-piece-transformer`
Commit: `5202d9a`
Kernel: `trydotatwo/cayley-beam-transformer-libtorch-2xt4-benchmark`, Kaggle version 2
Model source: `vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1`

Change under test:

- C++ LibTorch token construction precomputed active piece indices/positions per slot.
- Token build updated only active pieces through LibTorch advanced indexing / `index_put_` instead of dense gather+mask.
- Benchmark process used `c10::InferenceMode`.
- Batch sweep expanded to `[128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048]`.

Result:

| GPU | Best batch | Candidates/s |
|---:|---:|---:|
| 0 | 384 | 463779.0 |
| 1 | 320 | 453246.0 |
| aggregate | | 917025.0 |

Comparison:

| Baseline | Aggregate candidates/s | Ratio |
|---|---:|---:|
| LibTorch v1 dense token build | 947512.0 | 0.9689x |
| PyTorch batch_process reference | 1261394.0 | 0.7270x |
| native/CUTLASS v19 | 1224735.7 | 0.7492x |

Conclusion:

The active-index token-build fast path is a regression on 2xT4. The likely cause is LibTorch advanced indexing/scatter overhead exceeding the saved masked gather work for this small `num_pieces=50,max_piece_size=12` shape. The active-index code should not be kept in the benchmark scaffold. Keep `InferenceMode` and the broader sweep, then re-measure dense token build as v3.

Artifacts:

- `test_results/kaggle_libtorch_transformer_benchmark_v2_2026-07-04/stream1_libtorch_transformer_summary.json`
- `test_results/kaggle_libtorch_transformer_benchmark_v2_2026-07-04/stream1_libtorch_transformer_rows.csv`
- `test_results/kaggle_libtorch_transformer_benchmark_v2_2026-07-04/cayley-beam-transformer-libtorch-2xt4-benchmark.log`