# IHES bucket prebuilt array docs

Date: 2026-06-26

Scope:
- Updated the IHES bucket array launch documentation to build one shared `production_runner` via `prepare_ihes_prebuilt_runner.sh`.
- The documented array run now exports `BEAM_PREBUILT_RUNNER` and depends on the prebuild job with `--dependency=afterok:<job>`.

Verification:
- Existing `hpc/ihes_cube_model/ihes_solve_bucket_from_scratch.sh` support for `BEAM_PREBUILT_RUNNER` was checked.
- Existing `hpc/ihes_cube_model/prepare_ihes_prebuilt_runner.sh` builds `${RUN_DIR}/prebuilt-a100-ihes/production_runner` and prints `prebuilt_runner_ready=1`.

Architecture:
- No CUDA, solver, or runtime behavior changed.
