# Stream1 Piece-Transformer Kaggle 2xT4 Benchmark v8 FF1 Fused - 2026-07-03

## Source

- Kaggle kernel: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`, version 8.
- Branch: `codex/stream1-piece-transformer`.
- Commit: `3bf864a`.
- Hardware: Kaggle `NvidiaTeslaT4`, two Tesla T4 GPUs.
- Backend: native Stream1 piece-transformer fp16 path.
- Change under test: FF1 projection fused as CUTLASS GEMM + bias + SiLU.

Downloaded outputs:

- `test_results/kaggle_stream1_transformer_2xt4_v8_ff1_fused_2026-07-03/stream1_transformer_benchmark_rows.csv`
- `test_results/kaggle_stream1_transformer_2xt4_v8_ff1_fused_2026-07-03/cayley-beam-transformer-2xt4-benchmark.log`
- `test_results/kaggle_stream1_transformer_2xt4_v8_ff1_fused_2026-07-03/stream1_transformer_benchmark_reports/`

## Results

| version | gpu | best b_micro | concurrency | rows/group | ms/group | candidates/s | scratch bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 1024 | 2 | 2048 | 120.4175 | 408179.9 | 655458304 |
| 7 | 1 | 1024 | 2 | 2048 | 121.1607 | 405676.1 | 655458304 |
| 8 | 0 | 512 | 2 | 1024 | 55.0670 | 446292.4 | 327729152 |
| 8 | 1 | 1024 | 2 | 2048 | 109.5028 | 448865.3 | 655458304 |

## Interpretation

- GPU0 improved from `408179.9` to `446292.4` candidates/s (`1.093x`).
- GPU1 improved from `405676.1` to `448865.3` candidates/s (`1.106x`).
- The v8 best point is still below the PyTorch transformer reference:
  - PyTorch direct forward best: `734786.8` candidates/s.
  - PyTorch `batch_process`: `630697.0` candidates/s.
- FF1 fusion helps, but the remaining gap is still larger than a single epilogue. The next likely bottlenecks are the remaining transformer epilogues/layout transitions and the T4 attention path.

## Status

- Kaggle log reports branch `codex/stream1-piece-transformer` and commit `3bf864a`.
- Both GPU benchmark reports ended with `status=pass`.
- Grep over the Kaggle log found no traceback/error/failure lines in the bounded check.