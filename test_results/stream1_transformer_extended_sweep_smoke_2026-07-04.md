# Stream1 Transformer Extended Sweep Smoke

Date: 2026-07-04

Changed transformer benchmark sweep to cover more T4-relevant points before doing deeper CUDA rewrites:

```text
B_MICRO:     256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096
concurrency: 1, 2, 3, 4, 5, 6
```

This is benchmark-only and does not change production Stream1 inference.

Docker smoke:

```text
BEAM_STREAM1_TRANSFORMER_BLOCK51=1
BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1
BEAM_STREAM1_TRANSFORMER_B_MICRO=384
BEAM_STREAM1_TRANSFORMER_CONCURRENCY=3
stream1_transformer_micro b_micro=384 concurrency=3 rows_per_launch_group=1152 ms_per_launch_group=246.6505 parents_per_sec=4670.6 candidates_per_sec=112093.8 scratch_bytes=368695296
stream1_transformer_benchmark_done=1
```

The local GPU was not used for performance decisions here; this only validates the expanded harness before Kaggle T4 measurement.
