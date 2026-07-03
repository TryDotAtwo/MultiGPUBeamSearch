# Stream1 Piece Transformer Kaggle 2xT4 v11 CUTLASS FMHA Benchmark

## Scope

Validate the PyTorch-like CUTLASS FMHA attention route on Kaggle 2xT4.

- Kernel: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`
- Kaggle version: `11`
- Source ref: `stream1-transformer-fmha-44352f1`
- Expected commit prefix: `44352f1`
- Runtime mode: `BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1`
- Status: `KernelWorkerStatus.COMPLETE`

## Output Location

Downloaded Kaggle artifacts:

- `test_results/kaggle_stream1_transformer_2xt4_v11_fmha_2026-07-03/`
- Main log: `test_results/kaggle_stream1_transformer_2xt4_v11_fmha_2026-07-03/cayley-beam-transformer-2xt4-benchmark.log`
- CSV: `test_results/kaggle_stream1_transformer_2xt4_v11_fmha_2026-07-03/stream1_transformer_benchmark_rows.csv`
- Per-GPU reports: `test_results/kaggle_stream1_transformer_2xt4_v11_fmha_2026-07-03/stream1_transformer_benchmark_reports/`

## Results

Best rows:

- GPU0: `b_micro=512`, `concurrency=2`, `candidates_per_sec=502218.5`, `scratch_bytes=327729152`
- GPU1: `b_micro=512`, `concurrency=2`, `candidates_per_sec=500790.0`, `scratch_bytes=327729152`
- Aggregate: `STREAM1_TRANSFORMER_BEST_2XT4_AGG_CANDIDATES_PER_SEC=1003008.5`

## Comparison

Previous Kaggle v10 graph replay route:

- GPU0: `468639.3` candidates/s
- GPU1: `462338.7` candidates/s
- Aggregate: `930978.0` candidates/s

FMHA v11 is therefore about:

- GPU0: `1.072x` of v10
- GPU1: `1.083x` of v10
- Aggregate: `1.077x` of v10

PyTorch reference from the transformer inference example was about `630697.0` candidates/s per T4 in searcher-like `batch_process` mode, so v11 reaches roughly `0.80x` of PyTorch per T4.

## Interpretation

The fused CUTLASS FMHA route is a real T4 improvement over the previous decomposed native attention path, but it still trails PyTorch SDPA. The most visible remaining difference is that v11 still repacks QKV into a contiguous FMHA layout before attention; PyTorch SDPA can consume strided Q/K/V views.