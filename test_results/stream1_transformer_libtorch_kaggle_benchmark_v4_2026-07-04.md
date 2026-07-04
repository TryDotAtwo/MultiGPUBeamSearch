# Stream1 LibTorch Transformer Kaggle 2xT4 Benchmark v4

Date: 2026-07-04

Branch: `codex/stream1-piece-transformer`
Commit: `8a8cb5f`
Kernel: `trydotatwo/cayley-beam-transformer-libtorch-2xt4-benchmark`, Kaggle version 4
Model source: `vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1`

Change under test:

- Removed explicit `.contiguous()` copies from Q/K/V tensors before LibTorch `scaled_dot_product_attention`.
- Kept dense token build and `c10::InferenceMode`.
- Kept broader batch sweep `[128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048]`.

Result:

| GPU | Best batch | Candidates/s |
|---:|---:|---:|
| 0 | 320 | 513315.0 |
| 1 | 320 | 495637.0 |
| aggregate | | 1008952.0 |

Comparison:

| Baseline | Aggregate candidates/s | Ratio |
|---|---:|---:|
| LibTorch v1 dense token build | 947512.0 | 1.0648x |
| LibTorch v3 dense + InferenceMode | 971608.0 | 1.0384x |
| PyTorch batch_process reference | 1261394.0 | 0.7999x |
| native/CUTLASS v19 | 1224735.7 | 0.8238x |

Conclusion:

Removing forced Q/K/V contiguous copies is a real win on 2xT4. The SDPA path handles the permuted Q/K/V views better than paying explicit copies in C++ for this shape. Keep this change unless a later production integration shows graph-capture constraints.

Artifacts:

- `test_results/kaggle_libtorch_transformer_benchmark_v4_2026-07-04/stream1_libtorch_transformer_summary.json`
- `test_results/kaggle_libtorch_transformer_benchmark_v4_2026-07-04/stream1_libtorch_transformer_rows.csv`
- `test_results/kaggle_libtorch_transformer_benchmark_v4_2026-07-04/cayley-beam-transformer-libtorch-2xt4-benchmark.log`