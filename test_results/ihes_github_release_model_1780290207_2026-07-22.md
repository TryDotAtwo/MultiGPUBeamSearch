# IHES GitHub Release Model Update

Date: 2026-07-22

Scope:
- Preserve the old IHES cluster launch path: prepare from GitHub release,
  prebuild one `production_runner`, then launch
  `ihes_solve_bucket_from_scratch.sh` through SLURM.
- Switch the default IHES model asset to
  `p888-t000_1780290207_e40960.pth`, matching Kaggle dataset
  `arabidopsisthalian/model-ihes-1780290207-e40960`.

Tracked changes:
- `hpc/ihes_cube_model/prepare_ihes_cube_model.sh`
  - default `MODEL_RELEASE_ASSET` is now `p888-t000_1780290207_e40960.pth`;
  - default `model.pth` uses a `model.pth.release_asset` marker so a changed
    release asset replaces stale cluster-side weights.
- `hpc/ihes_cube_model/README.md`
  - documents the new default asset and marker behavior.

Release work:
- Uploaded `p888-t000_1780290207_e40960.pth` to GitHub release `ihes-p888-model`.
- Release digest: `sha256:0e29248fa03ba3f522e015fa5ab8634171d152c4e2a4fc2d9f0b787982cc34d9`.
- Verified existing release data asset `cayleypy-ihes-cube.zip` contains ids `1001` and `1002`.
- The model file is intentionally not committed to git.

Validation:
- `git diff --check` passed for tracked edits.
- GitHub release asset upload completed.
- Local `bash -n` was not available: Windows bash/Git Bash failed with access-denied errors in this sandbox.
