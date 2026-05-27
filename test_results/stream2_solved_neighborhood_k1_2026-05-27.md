# Stream2 Solved-Neighborhood K1 Verification 2026-05-27

## Scope
- Implemented `BEAM_SOLVED_NEIGHBORHOOD_RADIUS`.
- Radius `0` keeps the existing Stream2 exact central-state check.
- Radius `>0` enables CPU-built inverse solved-neighborhood lookup.
- GPU table stores only fingerprints and full `Hash128` slots.
- CPU stores `Hash128 -> packed suffix` and appends the suffix after history-prefix reconstruction.
- K2 descendant expansion was documented only, not implemented.

## Implementation Notes
- `src/hash.hpp`: added reusable `Hash128` fingerprint and two bucket-key helpers.
- `cuda/stream2.hpp`: added `SolvedNeighborhoodDeviceTable`.
- `cuda/stream2.cu`: added exact two-bucket lookup with fingerprint prefilter and full-hash confirmation.
- `tools/production_runner.cu`: added CPU neighborhood builder, suffix map, runner config logs, best solved-hit selection by total depth, global multi-rank solved-hit selection by total depth, and suffix append after reconstruction.
- `kaggle/*.ipynb`: added `SOLVED_NEIGHBORHOOD_RADIUS` and `SOLVED_NEIGHBORHOOD_MAX_ENTRIES`.
- `ARCHITECTURE_NEED.md`: documented K1 and future K2.

## Verification
- `git diff --check`: pass.
- Docker build: `production_runner stream2_cuda_tests dispatcher_cuda_tests static_memory_cuda_tests`: pass.
- Docker tests: `stream2_cuda_tests`, `dispatcher_cuda_tests`, `static_memory_cuda_tests`: pass.
- Notebook JSON parse: `kaggle/cayley-beam-gpu-runner.ipynb`, `kaggle/beam_kernel.ipynb`: pass.
- Small production smoke:
  - command: `production_runner 0 1 4096`
  - env: `BEAM_SOLVED_NEIGHBORHOOD_RADIUS=1`, `BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES=100000`, manual low-memory runtime config
  - result: pass, no crash, `solved_neighborhood_entries=25`, `solved_neighborhood_bucket_count=32`, `solved_neighborhood_device_bytes=2560`

## Residual Notes
- Current Stream2 stop behavior remains the existing solved-stop path.
- Multi-hit full-depth continuation and K2 descendant expansion remain future work requiring separate approval.
