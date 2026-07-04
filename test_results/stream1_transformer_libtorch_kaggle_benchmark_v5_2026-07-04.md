# Stream1 LibTorch Transformer Kaggle 2xT4 Benchmark v5

Date: 2026-07-04

Branch: `codex/stream1-piece-transformer`
Commit: `7487afb`
Kernel: `trydotatwo/cayley-beam-transformer-libtorch-2xt4-benchmark`, Kaggle version 5
Model source: `vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1`

Change under test:

- Replaced C++ `torch::matmul(x, weight_hxk) + bias` projections with `at::linear(x, weight_hxk.transpose(0, 1), bias)`.
- Applied to QKV, attention output, FF1, FF2, and final output projections.
- Kept no-QKV-contiguous SDPA, dense token build, and `c10::InferenceMode`.

Result:

| GPU | Best batch | Candidates/s |
|---:|---:|---:|
| 0 | 384 | 665466.0 |
| 1 | 448 | 669009.0 |
| aggregate | | 1334475.0 |

Comparison:

| Baseline | Aggregate candidates/s | Ratio |
|---|---:|---:|
| LibTorch v1 dense token build | 947512.0 | 1.4084x |
| LibTorch v4 no QKV contiguous | 1008952.0 | 1.3226x |
| PyTorch batch_process reference | 1261394.0 | 1.0579x |
| native/CUTLASS v19 | 1224735.7 | 1.0896x |

Conclusion:

`at::linear` is the major missing C++/PyTorch dispatch match. With dense token build, no explicit Q/K/V contiguous copies, `c10::InferenceMode`, and `at::linear`, the C++ LibTorch scaffold now beats the recorded PyTorch `batch_process` 2xT4 reference by about 5.8% and native/CUTLASS v19 by about 9.0% on this benchmark.

The score-key checksums are not bit-identical to the older `matmul + bias` C++ path, which is expected from a different FP16 projection kernel/order. Production integration should include an explicit tolerance check against the Python reference on fixed states before routing solver Stream1 through this backend.

Default no-LibTorch contract_tests=pass was re-run after the v5 code change in gpu-dev-cutlass-nsight:cuda128-sm120, confirming the default/MLP build remains Torch-free and unaffected.

Artifacts:

- `test_results/kaggle_libtorch_transformer_benchmark_v5_2026-07-04/stream1_libtorch_transformer_summary.json`
- `test_results/kaggle_libtorch_transformer_benchmark_v5_2026-07-04/stream1_libtorch_transformer_rows.csv`
- `test_results/kaggle_libtorch_transformer_benchmark_v5_2026-07-04/cayley-beam-transformer-libtorch-2xt4-benchmark.log`