# Stream1 Piece-Transformer Kaggle 2xT4 Benchmark v9 QKV Fused - 2026-07-03

## Source

- Kaggle kernel: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`, version 9.
- Branch: `codex/stream1-piece-transformer`.
- Commit: `83c95ea`.
- Hardware: Kaggle `NvidiaTeslaT4`, two Tesla T4 GPUs.
- Backend: native Stream1 piece-transformer fp16 path.
- Change under test: QKV projection fused as CUTLASS GEMM + bias, on top of the previous FF1 GEMM + bias + SiLU fusion.

Downloaded outputs:

- `test_results/kaggle_stream1_transformer_2xt4_v9_qkv_fused_2026-07-03/stream1_transformer_benchmark_rows.csv`
- `test_results/kaggle_stream1_transformer_2xt4_v9_qkv_fused_2026-07-03/cayley-beam-transformer-2xt4-benchmark.log`
- `test_results/kaggle_stream1_transformer_2xt4_v9_qkv_fused_2026-07-03/stream1_transformer_benchmark_reports/`

## Best Points

| version | change | gpu | best b_micro | concurrency | rows/group | ms/group | candidates/s | scratch bytes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 7 | clean tensor attention baseline | 0 | 1024 | 2 | 2048 | 120.4175 | 408179.9 | 655458304 |
| 7 | clean tensor attention baseline | 1 | 1024 | 2 | 2048 | 121.1607 | 405676.1 | 655458304 |
| 8 | FF1 fused | 0 | 512 | 2 | 1024 | 55.0670 | 446292.4 | 327729152 |
| 8 | FF1 fused | 1 | 1024 | 2 | 2048 | 109.5028 | 448865.3 | 655458304 |
| 9 | FF1 + QKV fused | 0 | 1024 | 2 | 2048 | 103.6711 | 474114.7 | 655458304 |
| 9 | FF1 + QKV fused | 1 | 512 | 2 | 1024 | 52.5856 | 467352.5 | 327729152 |

## Interpretation

- v9 vs v8:
  - GPU0: `446292.4 -> 474114.7` candidates/s (`1.062x`).
  - GPU1: `448865.3 -> 467352.5` candidates/s (`1.041x`).
- v9 vs v7 baseline:
  - GPU0: `408179.9 -> 474114.7` candidates/s (`1.162x`).
  - GPU1: `405676.1 -> 467352.5` candidates/s (`1.152x`).
- v9 remains below PyTorch transformer reference:
  - PyTorch direct forward best: `734786.8` candidates/s.
  - PyTorch `batch_process`: `630697.0` candidates/s.
- The QKV bias epilogue is a real improvement, but the remaining gap is still in the transformer attention/layout/remaining epilogues rather than FF1/QKV bias alone.

## Status

- Kaggle log reports branch `codex/stream1-piece-transformer` and commit `83c95ea`.
- Both GPU benchmark reports ended with `status=pass`.
- Bounded grep over the Kaggle log found no traceback/error/failure lines.