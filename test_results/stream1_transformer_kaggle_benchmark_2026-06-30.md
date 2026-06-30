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
