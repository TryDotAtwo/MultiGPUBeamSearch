# Stream1 Piece-Transformer Kaggle 2xT4 Benchmark

Date: 2026-06-30
Branch: `codex/stream1-piece-transformer`

## Scope

Created and launched a private Kaggle package for measuring the split `stream_benchmark` transformer backend on a 2xT4 runtime.

Package: `kaggle_t4_transformer_benchmark/`
Kernel id: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`

## Local Validation Before Push

- `kernel-metadata.json`: JSON parse passed.
- `t4-transformer-stream1-benchmark.ipynb`: JSON parse passed.
- Notebook code cells: Python AST parse passed.

## Benchmark Method

- Clone GitHub branch `codex/stream1-piece-transformer`.
- Export Kaggle model `vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1` to fp16 `piece_transformer` weights.
- Build only `stream_benchmark` for `sm_75`.
- Run `stream_benchmark` once with `CUDA_VISIBLE_DEVICES=0` and once with `CUDA_VISIBLE_DEVICES=1`.
- Parse `stream1_transformer_micro` rows and write `/kaggle/working/stream1_transformer_benchmark_rows.csv`.

## Remote Result

Pending Kaggle execution.

## Remote Result V1

- Kernel version: 1.
- Status: `KernelWorkerStatus.ERROR`.
- Output directory: `test_results/kaggle_t4_transformer_benchmark_v1_output/`.
- Failure: notebook preflight still used copied smoke `derived_values()` and expected `TORCHRUN_NNODES`, but the benchmark config cell had removed torchrun variables. Fixed in the benchmark notebook by keeping the smoke solver config variables for shared preflight sanity checks while still building/running only `stream_benchmark`.

## Remote Result V2

- Kernel version: 2.
- Status: `KernelWorkerStatus.COMPLETE`.
- Output directory: `test_results/kaggle_t4_transformer_benchmark_v2_output/`.
- Parsed rows: `30` (`15` configs on GPU0 and `15` configs on GPU1).
- CSV: `test_results/kaggle_t4_transformer_benchmark_v2_output/stream1_transformer_benchmark_rows.csv`.
- Logs:
  - `test_results/kaggle_t4_transformer_benchmark_v2_output/stream1_transformer_benchmark_logs/stream1_transformer_benchmark_gpu0.log`
  - `test_results/kaggle_t4_transformer_benchmark_v2_output/stream1_transformer_benchmark_logs/stream1_transformer_benchmark_gpu1.log`
  - `test_results/kaggle_t4_transformer_benchmark_v2_output/cayley-beam-transformer-2xt4-benchmark.log`

Best rows:

| GPU | B_MICRO | concurrency | rows/group | ms/group | parents/s | candidates/s | scratch bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 512 | 1 | 512 | 108.0227 | 4739.7 | 113753.8 | 162963456 |
| 1 | 512 | 1 | 512 | 103.2894 | 4956.9 | 118966.7 | 162963456 |

Estimated best aggregate for two independent T4 ranks: `232720.5` candidates/s, `9696.6` parents/s.

Observation: increasing either `B_MICRO` or `concurrency` reduces throughput on T4 for the current transformer backend. Current best is the smallest tested config, `B_MICRO=512`, `STREAM1_CONCURRENCY=1`.
