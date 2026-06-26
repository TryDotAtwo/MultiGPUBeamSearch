# IHES bucket public results publish wrapper

Date: 2026-06-26

Scope:
- Added optional publishing to `hpc/ihes_cube_model/ihes_solve_bucket_from_scratch.sh`.
- No CUDA kernels, solver architecture, Stream 2/3/4/5 logic, or runner internals changed.
- Each completed puzzle can export per-puzzle TSV/metadata/summary files to a separate GitHub results repository.

Verification:
- `docker run --rm -v D:\100XH100:/work -w /work gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "bash -n hpc/ihes_cube_model/ihes_solve_bucket_from_scratch.sh"` passed.

Notes:
- Publishing is enabled by setting `PUBLISH_RESULTS_REPO_URL` or by precreating `PUBLISH_RESULTS_DIR` as a git checkout.
- `PUZZLE_LIMIT=1` is intentional for SLURM arrays: one task processes one selected puzzle.
- The publish step is best-effort and does not fail the completed compute run if clone/push is unavailable.
