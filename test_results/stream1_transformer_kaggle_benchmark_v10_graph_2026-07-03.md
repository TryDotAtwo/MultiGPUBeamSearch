# Stream1 Piece Transformer Kaggle 2xT4 v10 Graph Benchmark

## Scope

- Kernel: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`
- Kaggle version: `10`
- Source ref: `stream1-transformer-graph-bench-e35432d`
- Expected commit prefix: `e35432d`
- Notebook package: `kaggle_t4_transformer_benchmark/`
- Runtime mode: `BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1`
- Status: `KernelWorkerStatus.COMPLETE`

This run only validates the graph-replay benchmark path on Kaggle 2xT4. No CUDA source or MLP-path behavior was changed for this launch.

## Local Preflight

- `kernel-metadata.json` inspected: private notebook, `NvidiaTeslaT4`, model source `vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1`.
- Notebook JSON parse: pass.
- Notebook Python code-cell AST parse: pass, 3 code cells.
- Notebook config confirms:
  - `GITHUB_REF = "stream1-transformer-graph-bench-e35432d"`
  - `GITHUB_EXPECTED_COMMIT = "e35432d"`
  - per-GPU benchmark env includes `BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1`

## Output Location

Downloaded Kaggle artifacts:

- `test_results/kaggle_stream1_transformer_2xt4_v10_graph_2026-07-03/`
- Main log: `test_results/kaggle_stream1_transformer_2xt4_v10_graph_2026-07-03/cayley-beam-transformer-2xt4-benchmark.log`
- CSV: `test_results/kaggle_stream1_transformer_2xt4_v10_graph_2026-07-03/stream1_transformer_benchmark_rows.csv`
- Per-GPU reports: `test_results/kaggle_stream1_transformer_2xt4_v10_graph_2026-07-03/stream1_transformer_benchmark_reports/`

## Results

Best lines from the Kaggle log:

- GPU0: `b_micro=1024`, `concurrency=1`, `candidates_per_sec=468639.3`, `scratch_bytes=327729152`
- GPU1: `b_micro=512`, `concurrency=2`, `candidates_per_sec=462338.7`, `scratch_bytes=327729152`
- Aggregate: `STREAM1_TRANSFORMER_BEST_2XT4_AGG_CANDIDATES_PER_SEC=930978.0`

The per-report files both show `graph_bench=1` and `status=pass`.

## Comparison

Previous Kaggle v9 QKV-fused benchmark:

- GPU0: `474114.7` candidates/s
- GPU1: `467352.5` candidates/s
- Aggregate: `941467.2` candidates/s

Graph replay v10 is therefore about:

- GPU0: `0.988x` of v9
- GPU1: `0.989x` of v9
- Aggregate: `0.989x` of v9

PyTorch reference from the transformer inference example was about `630697.0` candidates/s per T4 in the searcher-like `batch_process` mode, so v10 native graph replay reaches about `0.74x` / `0.73x` of that reference.

## Interpretation

CUDA graph replay on Kaggle T4 does not materially improve the isolated Stream1 transformer benchmark. This matches the local Nsight finding: launch overhead is no longer the main bottleneck under graph replay. The remaining gap is still in transformer linear GEMMs, LayerNorm/copy work, attention layout, and remaining unfused epilogues.
