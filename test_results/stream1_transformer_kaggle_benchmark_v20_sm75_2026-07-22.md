# Stream1 transformer Kaggle 2xT4 v20 SM75 baseline (2026-07-22)

## Scope

Establish a hardware-specific SM75 baseline after closing the local RTX 3070 / SM86 contour.

- Private kernel: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`
- Kaggle version: 20
- Status: `KernelWorkerStatus.COMPLETE`
- Source tag: `stream1-transformer-rtx3070-final-18e5996`
- Checked-out commit: `18e5996`
- GPUs: two NVIDIA T4, measured independently
- Mode: native FP16 block51 CUDA Graph

## Exactness

The run produced 80 complete score dumps: 8 policy configurations x 5 independent processes x 2 GPUs.

All 80 files independently hash to:

`6524c19ff92c7c87263eb9c1e3d2d64ccd20c6ca117e803e6b44770f226644ef`

This includes full-final, final-CLS-only, Q=1 CLS attention, full-attention tile/max-K/alignment variants, and persistent LayerNorm. There were no policy errors.

## Shape sweep

Two processes per point selected the same shape on both T4s:

| GPU | Best shape | Median candidates/s |
|---:|---|---:|
| 0 | `b_micro=384, concurrency=1` | 749,260.0 |
| 1 | `b_micro=384, concurrency=1` | 749,477.4 |

Concurrency 2 was slower at every comparable point. The best prior v19 hardware points were not reused as assumptions.

## Policy medians at 384 x 1

| Policy | GPU0 | GPU1 | Aggregate | Aggregate vs final-CLS baseline |
|---|---:|---:|---:|---:|
| Full final layer | 612,923.1 | 603,423.4 | 1,216,346.5 | 0.8162x |
| Final CLS only, full final attention | 749,458.1 | 740,741.4 | 1,490,199.5 | 1.0000x |
| Final CLS Q=1 q64 | 749,957.3 | 745,406.6 | 1,495,363.9 | 1.0035x |
| Final CLS Q=1 q32 | 756,128.8 | 746,952.2 | 1,503,081.0 | 1.0086x |
| Full attention q32k64 | 728,219.8 | 721,535.1 | 1,449,754.9 | 0.9729x |
| Full attention exact32 | 745,186.4 | 736,739.7 | 1,481,926.1 | 0.9944x |
| Full attention q64k64v4 | 743,067.9 | 728,192.5 | 1,471,260.4 | 0.9873x |
| Persistent LayerNorm | 747,818.6 | 727,051.8 | 1,474,870.4 | 0.9897x |

Final-CLS-only is a confirmed exact T4 win: aggregate throughput is 1.2251x full-final and 1.2168x the older v19 aggregate of 1,224,735.7 candidates/s. Q=1 q32 is exact but improves the selected final-CLS baseline by only 0.86%, below the 3% acceptance gate. All other policy candidates regress or are neutral.

## Decision

Select `b_micro=384, concurrency=1` and final-CLS-only with full final attention as the current SM75 baseline. Do not select Q=1, q32 full attention, exact32, v4 alignment, or persistent LayerNorm.

The next measured boundary is SM75-specific GEMM policy support. The current source exposes SM80 GEMM autotuning but keeps SM75 QKV, residual, and FF1 tiles hard-coded, so T4 cannot yet measure the same family-level tile/stage/swizzle axes.

## Artifacts

- `test_results/kaggle_stream1_transformer_2xt4_v20_sm75_2026-07-22/cayley-beam-transformer-2xt4-benchmark.log`
- `test_results/kaggle_stream1_transformer_2xt4_v20_sm75_2026-07-22/stream1_transformer_t4_summary.json`
- `test_results/kaggle_stream1_transformer_2xt4_v20_sm75_2026-07-22/stream1_transformer_t4_rows.csv`
- `test_results/kaggle_stream1_transformer_2xt4_v20_sm75_2026-07-22/stream1_transformer_score_dumps/`
