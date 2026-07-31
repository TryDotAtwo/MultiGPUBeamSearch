
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
- Full suite and real Kaggle 2xT4 acceptance are recorded below after completion.
