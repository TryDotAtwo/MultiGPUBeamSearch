
# Kaggle 2xT4 final-materialization profile policy

Date: 2026-07-31

## Requirement

Use `final_materialize_chunk_candidates=88,064` explicitly across every public Kaggle 2xT4 profile, for both MLP families and Piece Transformer, including small beam anchors.

## Scope

- MLP output1 anchors 2^16 through 2^25: 10 profiles.
- MLP output-move-count anchors 2^16 through 2^25: 10 profiles.
- Piece Transformer output-move-count anchors 2^16 through 2^26: 11 profiles.
- Total: 31 explicit profile records.

The change affects configuration only. It does not alter CUDA/C++ beam-search architecture or semantics.

## Verification

- Registry-wide invariant test added in `tests/cayleypy_public/test_profile.py`.
- Focused profile/runner suite: `66 passed`.
- Full public/exporter suite: `261 passed`.
- Private Kaggle acceptance `trydotatwo/cayleypy-2xt4-universal-cube4-acceptance` v11: `COMPLETE` on two Tesla T4 GPUs.
- Workload: Piece Transformer output-24, requested/effective beam `2^16`, depth 8, puzzle 0.
- Selected profile: anchor 16 with explicit `final_materialize_chunk_candidates=88,064`; both ranks returned 0.
- Rank depth processing: 6.190 s / 6.188 s; solve orchestration: 14.464 s; no OOM, overflow, or fatal error.
- Evidence directory: `D:/100XH100/test_results/kaggle_cube4_universal_acceptance_v11`.
- Public notebook `trydotatwo/cayleypy-2xt4-checkpoint-beam-search` v5: `COMPLETE`, clean `SETUP_REQUIRED` landing, solver pin `a1db0e6d9bb5458c8a842b37dfa99572d3025667`.
- Public log: `D:/100XH100/test_results/kaggle_cayleypy_public_v5/cayleypy-2xt4-checkpoint-beam-search.log`.
