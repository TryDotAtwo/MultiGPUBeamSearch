# Stream1 Transformer Kaggle 2xT4 Benchmark v17 - Block51

Date: 2026-07-04
Kaggle kernel: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`
Kaggle version: 17
Git ref: `stream1-transformer-block51-0d86115`
Git commit: `0d86115`

## Result

The exact-shape `block51` backend compiled and ran successfully on Kaggle 2xT4.
It is correct as a runnable backend, but it is not a T4 speed win as the default
path.

Best rows:

```text
gpu0: b_micro=512 concurrency=1 candidates_per_sec=598701.1 scratch_bytes=163864576
gpu1: b_micro=512 concurrency=2 candidates_per_sec=592691.2 scratch_bytes=327729152
aggregate_2xt4_candidates_per_sec=1191392.3
```

Comparison against earlier T4 runs:

```text
v15 biasadd256 aggregate:      1213776.7 candidates/s
v16 bias+LayerNorm aggregate:  1210546.2 candidates/s
v17 block51 aggregate:         1191392.3 candidates/s
v17 / v15:                     0.9816x (-1.84%)
v17 / v16:                     0.9842x (-1.58%)
```

## Decision

Do not leave `block51` auto-routed as the default exact-shape path yet. The code
is kept as the first shape-specialized transformer block foundation, but the
runtime route should be explicit opt-in via `BEAM_STREAM1_TRANSFORMER_BLOCK51=1`.
If that variable is set for a non-matching shape, the backend must throw instead
of silently falling back.

## Follow-up Target

The first fused input+LayerNorm stage removed one graph stage but did not beat
the measured T4 `biasadd256` schedule. The next useful block51-specific work is
not more launch fusion around input; it is one of:

- specialized LayerNorm + QKV projection for `rows=b_micro*51`, `K=256`, `N=768`;
- CUTLASS tile/kernel selection for QKV, FF1, FF2, and residual projections on SM75;
- a custom residual+bias epilogue path that avoids the slower broadcast GEMM
  behavior seen in earlier profiling.
