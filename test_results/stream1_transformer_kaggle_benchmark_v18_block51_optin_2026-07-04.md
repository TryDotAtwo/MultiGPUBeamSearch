# Stream1 Transformer Kaggle 2xT4 Benchmark v18 - Opt-in Block51

Date: 2026-07-04
Kaggle kernel: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`
Kaggle version: 18
Git ref: `stream1-transformer-block51-optin-7adbe14`
Git commit: `7adbe14`
Notebook env: `BEAM_STREAM1_TRANSFORMER_BLOCK51=1`

## Result

The final opt-in `block51` route compiled and ran successfully on Kaggle 2xT4.
This validates the final implementation shape: the backend is explicit, not an
automatic exact-shape route.

Best rows:

```text
gpu0: b_micro=1024 concurrency=1 candidates_per_sec=610697.3 scratch_bytes=327729152
gpu1: b_micro=512  concurrency=1 candidates_per_sec=601347.2 scratch_bytes=163864576
aggregate_2xt4_candidates_per_sec=1212044.5
```

Comparison:

```text
v15 biasadd256 aggregate:      1213776.7 candidates/s
v16 bias+LayerNorm aggregate:  1210546.2 candidates/s
v17 auto block51 aggregate:    1191392.3 candidates/s
v18 opt-in block51 aggregate:  1212044.5 candidates/s
v18 / v15:                     0.9986x (-0.14%)
v18 / v16:                     1.0012x (+0.12%)
v18 / v17:                     1.0173x (+1.73%)
```

## Decision

Keep `block51` as an explicit backend foundation, not the default. It is now
close to the best measured T4 path, but still does not beat v15 `biasadd256`.
The default path should stay conservative until a block-specialized projection or
epilogue actually wins on T4.
