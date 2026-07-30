# Public CayleyPy result-mode publication — 2026-07-30

- Requirement: publish one best result per puzzle in `first`; publish every validated discovered result in `collect`.
- `first` selection: shortest `original_oriented_path`, with deterministic path/variant/depth tie-breaking.
- `collect` selection: all valid non-source solution records retained in deterministic runner order.
- `submission.csv`: unchanged best-per-puzzle behavior in both modes.
- Delivery: existing canonical envelopes and bounded batches (at most 100 results and 4 MiB per request).
- Beam/CUDA architecture: unchanged.
- Public solver pin: `737b25667a9a3d90645bc80351ea48f7fd3a3c1f`.
## Verification

- TDD red: both mode-selection tests failed with missing `_publication_records`.
- TDD green: focused tests passed 2/2.
- Full public suite: 240 passed.
- Notebook JSON/code-cell AST and metadata: valid, public, exact Kaggle slug.
- `git diff --check`: clean.
- Public artifact secret scan: clean.

