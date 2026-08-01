# Stream1 Piece Transformer Kaggle 2xT4 v15 BiasAdd256 Benchmark

## Scope

Validate the transformer-only fp16 `cols == 256` bias-add fast path on Kaggle 2xT4.

- Kernel: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`
- Kaggle version: `15`
- Source ref: `stream1-transformer-biasadd256-b34bf7c`
- Expected commit prefix: `b34bf7c`
- Runtime mode: `BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1`
- Status: `KernelWorkerStatus.COMPLETE`

The main Kaggle log confirms:

```text
GITHUB_REF= stream1-transformer-biasadd256-b34bf7c
GITHUB_COMMIT= b34bf7c
```

## Output Location

Downloaded Kaggle artifacts:

- `test_results/kaggle_stream1_transformer_2xt4_v15_biasadd256_2026-07-03/`

## Results

Best rows:

- GPU0: `b_micro=512`, `concurrency=2`, `candidates_per_sec=614318.7`, `scratch_bytes=327729152`
- GPU1: `b_micro=512`, `concurrency=4`, `candidates_per_sec=599458.0`, `scratch_bytes=655458304`
- Aggregate: `STREAM1_TRANSFORMER_BEST_2XT4_AGG_CANDIDATES_PER_SEC=1213776.7`

## Comparison To v14 LN128

Kaggle v14 LN128:

- GPU0: `593837.5` candidates/s
- GPU1: `545888.4` candidates/s
- Aggregate: `1139725.9` candidates/s

BiasAdd256 v15 is therefore about:

- GPU0: `1.034x` of v14
- GPU1: `1.098x` of v14
- Aggregate: `1.065x` of v14

## Comparison To PyTorch Reference

PyTorch reference from the transformer inference example was about `630697.0` candidates/s per T4 in searcher-like `batch_process` mode. v15 reaches roughly:

- GPU0: `0.974x` of PyTorch
- GPU1: `0.950x` of PyTorch
- Aggregate vs two T4 reference: `0.962x`

## Interpretation

The bias-add specialization is a real T4 win. Local Nsight already showed the standalone bias-add kernel share drop from about `7.0%` to `3.7%`; Kaggle v15 converts that into a `+6.5%` aggregate 2xT4 improvement over v14.

The backend is now close to the PyTorch transformer fast path on the measured Kaggle T4 benchmark, but it is still far slower than the native MLP Stream1 path because this transformer model performs multiple transformer blocks, attention, LayerNorm, and much larger linear work per candidate. Remaining native-vs-PyTorch gap is likely in GEMM/epilogue kernel choices, residual/bias fusion, and layout kernels rather than the old scalar attention implementation.