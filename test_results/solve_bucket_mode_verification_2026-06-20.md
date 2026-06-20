# Solve Bucket Mode Verification

Date: 2026-06-20

## Scope

- Added `BEAM_SOLVE_BUCKET_MODE` to keep ordinary solve-to-center runs alive
  after the first hit.
- Added `hpc/ihes_cube_model/ihes_solve_bucket.sh` for IHES length-23 original
  plus reflected runs.

## Checks

- Docker image: `gpu-dev-cutlass-nsight:cuda128-sm120`
- Build command:

```bash
cmake -S . -B build-solve-bucket-check -G Ninja -DCMAKE_BUILD_TYPE=Release -DBEAM_ENABLE_DEBUG=ON -DBEAM_ENABLE_DEPTH_LOGS=ON
cmake --build build-solve-bucket-check --target production_runner -j 8
```

Result: passed.

- Shell syntax:

```bash
bash -n hpc/ihes_cube_model/ihes_solve_bucket.sh
```

Result: passed.

- Local plan parse:

`hpc/ihes_cube_model/solv_uniq.csv` contains 318 puzzles whose best known
solution length is 23.

## Notes

The local Docker environment has no NVIDIA driver attached, so runtime GPU
execution was not attempted locally. Cluster runtime verification is still
required.
