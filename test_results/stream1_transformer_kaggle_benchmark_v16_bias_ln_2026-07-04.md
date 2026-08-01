# Stream1 Piece Transformer Kaggle 2xT4 v16 Bias+LayerNorm Benchmark

## Scope

Validate the transformer-only fused bias+LayerNorm path on Kaggle 2xT4.

- Kernel: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`
- Kaggle version: `16`
- Source ref: `stream1-transformer-bias-ln-65bf43d`
- Expected commit prefix: `65bf43d`
- Runtime mode: `BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1`
- Status: `KernelWorkerStatus.COMPLETE`

The main Kaggle log confirms:

```text
GITHUB_REF= stream1-transformer-bias-ln-65bf43d
GITHUB_COMMIT= 65bf43d
```

## Output Location

Downloaded Kaggle artifacts:

- `test_results/kaggle_stream1_transformer_2xt4_v16_bias_ln_2026-07-04/`

## Results

Best rows:

- GPU0: `b_micro=512`, `concurrency=1`, `candidates_per_sec=623236.4`, `scratch_bytes=163864576`
- GPU1: `b_micro=1024`, `concurrency=1`, `candidates_per_sec=587309.8`, `scratch_bytes=327729152`
- Aggregate: `STREAM1_TRANSFORMER_BEST_2XT4_AGG_CANDIDATES_PER_SEC=1210546.2`

## Comparison To v15 BiasAdd256

Kaggle v15 BiasAdd256:

- GPU0: `614318.7` candidates/s
- GPU1: `599458.0` candidates/s
- Aggregate: `1213776.7` candidates/s

Bias+LayerNorm v16 is therefore about:

- GPU0: `1.015x` of v15
- GPU1: `0.980x` of v15
- Aggregate: `0.997x` of v15

## Comparison To PyTorch Reference

PyTorch reference from the transformer inference example was about `630697.0` candidates/s per T4 in searcher-like `batch_process` mode. v16 reaches roughly:

- GPU0: `0.988x` of PyTorch
- GPU1: `0.931x` of PyTorch
- Aggregate vs two T4 reference: `0.960x`

## Interpretation

The fused bias+LayerNorm path is valid and locally improved SM86 graph replay,
but it is not a Kaggle T4 aggregate speed win. It removes standalone bias-add
graph nodes, yet the heavier fused LayerNorm kernels offset that on one T4 GPU.
For T4, v15 BiasAdd256 remains the better measured point by a small margin.

Next optimization work should focus on GEMM/epilogue kernel choices or attention
layout rather than more LayerNorm/bias plumbing.