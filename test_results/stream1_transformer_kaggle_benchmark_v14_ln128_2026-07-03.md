# Stream1 Piece Transformer Kaggle 2xT4 v14 LN128 Benchmark

## Scope

Validate the transformer-only LayerNorm128 optimization on Kaggle 2xT4.

- Kernel: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`
- Kaggle version: `14`
- Source ref: `stream1-transformer-ln128-0d32be5`
- Expected commit prefix: `0d32be5`
- Runtime mode: `BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1`
- Status: `KernelWorkerStatus.COMPLETE`

Version 13 was ignored because its notebook still cloned the old `stream1-transformer-strided-fmha-6d29ac4` tag. Version 14 is the valid LN128 run; its log confirms `GITHUB_REF=stream1-transformer-ln128-0d32be5` and `GITHUB_COMMIT=0d32be5`.

## Output Location

Downloaded Kaggle artifacts:

- `test_results/kaggle_stream1_transformer_2xt4_v14_ln128_2026-07-03/`
- Main log: `test_results/kaggle_stream1_transformer_2xt4_v14_ln128_2026-07-03/cayley-beam-transformer-2xt4-benchmark.log`
- CSV: `test_results/kaggle_stream1_transformer_2xt4_v14_ln128_2026-07-03/stream1_transformer_benchmark_rows.csv`
- Per-GPU reports: `test_results/kaggle_stream1_transformer_2xt4_v14_ln128_2026-07-03/stream1_transformer_benchmark_reports/`

## Results

Best rows:

- GPU0: `b_micro=512`, `concurrency=2`, `candidates_per_sec=593837.5`, `scratch_bytes=327729152`
- GPU1: `b_micro=512`, `concurrency=2`, `candidates_per_sec=545888.4`, `scratch_bytes=327729152`
- Aggregate: `STREAM1_TRANSFORMER_BEST_2XT4_AGG_CANDIDATES_PER_SEC=1139725.9`

## Comparison To v12 Strided FMHA

Kaggle v12 strided-QKV FMHA:

- GPU0: `573280.3` candidates/s
- GPU1: `560959.8` candidates/s
- Aggregate: `1134240.1` candidates/s

LN128 v14 is therefore about:

- GPU0: `1.036x` of v12
- GPU1: `0.973x` of v12
- Aggregate: `1.005x` of v12

## Comparison To PyTorch Reference

PyTorch reference from the transformer inference example was about `630697.0` candidates/s per T4 in searcher-like `batch_process` mode. v14 reaches roughly:

- GPU0: `0.942x` of PyTorch
- GPU1: `0.866x` of PyTorch
- Aggregate vs two T4 reference: `0.904x`

## Interpretation

The LayerNorm128 kernel is valid and locally/profile-wise useful: Nsight on the local graph profile reduced LayerNorm share from about `22.9%` to `11.3%`. On Kaggle T4, the end-to-end aggregate improvement is small (`+0.48%`) because the remaining wall time is dominated by CUTLASS GEMM/fused-epilogue groups and FMHA/kernel-choice differences, with noticeable GPU-to-GPU variance.

A residual+bias CUTLASS broadcast epilogue fusion was tested locally and rejected before commit because it regressed throughput. The next useful optimization should target GEMM/epilogue kernel choice or a custom residual+bias epilogue that does not switch to the slower broadcast GEMM path.