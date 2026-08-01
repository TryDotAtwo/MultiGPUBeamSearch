# Stream1 Piece Transformer Kaggle 2xT4 v12 Strided FMHA Benchmark

## Scope

Validate the strided-QKV CUTLASS FMHA route on Kaggle 2xT4.

- Kernel: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`
- Kaggle version: `12`
- Source ref: `stream1-transformer-strided-fmha-6d29ac4`
- Expected commit prefix: `6d29ac4`
- Runtime mode: `BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1`
- Status: `KernelWorkerStatus.COMPLETE`

## Output Location

Downloaded Kaggle artifacts:

- `test_results/kaggle_stream1_transformer_2xt4_v12_strided_fmha_2026-07-03/`
- Main log: `test_results/kaggle_stream1_transformer_2xt4_v12_strided_fmha_2026-07-03/cayley-beam-transformer-2xt4-benchmark.log`
- CSV: `test_results/kaggle_stream1_transformer_2xt4_v12_strided_fmha_2026-07-03/stream1_transformer_benchmark_rows.csv`
- Per-GPU reports: `test_results/kaggle_stream1_transformer_2xt4_v12_strided_fmha_2026-07-03/stream1_transformer_benchmark_reports/`

## Results

Best rows:

- GPU0: `b_micro=512`, `concurrency=2`, `candidates_per_sec=573280.3`, `scratch_bytes=327729152`
- GPU1: `b_micro=512`, `concurrency=2`, `candidates_per_sec=560959.8`, `scratch_bytes=327729152`
- Aggregate: `STREAM1_TRANSFORMER_BEST_2XT4_AGG_CANDIDATES_PER_SEC=1134240.1`

## Comparison

Kaggle v11 packed-QKV FMHA:

- GPU0: `502218.5` candidates/s
- GPU1: `500790.0` candidates/s
- Aggregate: `1003008.5` candidates/s

Strided-QKV v12 is therefore about:

- GPU0: `1.142x` of v11
- GPU1: `1.120x` of v11
- Aggregate: `1.131x` of v11

Previous v10 graph replay before FMHA:

- Aggregate: `930978.0` candidates/s

Strided-QKV v12 is `1.218x` of v10 aggregate.

PyTorch reference from the transformer inference example was about `630697.0` candidates/s per T4 in searcher-like `batch_process` mode. v12 reaches roughly:

- GPU0: `0.909x` of PyTorch
- GPU1: `0.889x` of PyTorch
- Aggregate vs two T4 reference: `0.899x`

## Interpretation

Removing the explicit QKV repack closes most of the previous native-vs-PyTorch SDPA gap. The remaining gap is no longer the old decomposed attention chain; it is likely in the remaining transformer kernels and CUTLASS FMHA/kernel choices versus PyTorch's tuned SDPA path.