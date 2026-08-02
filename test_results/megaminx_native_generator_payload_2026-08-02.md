# Megaminx native archive generator payload fix

Date: 2026-08-02

## Live evidence

Clean-cluster job 33340 initialized the corrected NCCL runtime but the first 30M probe failed because `FullBeamNice/generators/p900.json` was absent. The generic launcher error was also mislabeled `nccl_error` because the run directory contained the word `nccl`.

## Fix

- Stage `FullBeamNice/generators/p900.json` in every architecture archive.
- Require the file in the deterministic builder and independent archive checker.
- Add the exact path to the fixed public allowlist.
- Match concrete NCCL error phrases instead of any `nccl` substring.

## Verification

- Focused suite: 38 passed, 1 skipped.
- Full portable suite: 190 passed, 3 skipped.
- `git diff --check`: passed.

A new public release and clean-cluster rerun are required for runtime acceptance.