# Stream2 K2 Suffix Expansion Verification 2026-05-28

## Scope
- Added `BEAM_STREAM2_SUFFIX_RADIUS`.
- Added `BEAM_STREAM2_SUFFIX_BACKEND={base_generators,composed_permutations}`.
- Added optional `BEAM_STREAM2_SUFFIX_MAX_COUNT`.
- Added `solved_suffix_list[SOLVED_RESULT_CAPACITY]` in static device memory.
- K1 `BEAM_SOLVED_NEIGHBORHOOD_RADIUS` behavior remains unchanged.

## Architecture Result
- Stream2 still writes immediate `Hash128(parent + move)` into `hash_ring`.
- Stream2 checks K2 suffixes only when direct/K1 hit was not found and K2 is enabled.
- Stream2 records `solved_suffix_list[idx]=0` for direct/K1 hits.
- Stream2 records `solved_suffix_list[idx]=suffix_id` for K2 hits.
- CPU reconstruction appends K2 suffix first, then K1 suffix from `Hash128 -> PackedSuffix`.
- Distributed solved-hit packet now carries `suffix_id`; global best depth compares `prefix_depth + K2_suffix_len + K1_suffix_len`.

## Verification
- `git diff --check`: pass.
- Docker build targets: `stream2_cuda_tests`, `static_memory_cuda_tests`, `dispatcher_cuda_tests`, `production_runner`: pass.
- Docker targeted tests: `stream2_cuda_tests`, `static_memory_cuda_tests`, `dispatcher_cuda_tests`: pass.
- Docker post-guard rebuild targets: `stream2_cuda_tests`, `dispatcher_cuda_tests`, `production_runner`: pass.
- Docker post-guard test: `stream2_cuda_tests`: pass.
- Notebook JSON parse: `kaggle/beam_kernel.ipynb`: pass.
- Notebook JSON parse: `kaggle/cayley-beam-gpu-runner.ipynb`: pass.

## Notes
- K2 default is disabled through `STREAM2_SUFFIX_RADIUS=0` in both Kaggle notebooks.
- Existing K1 notebook radius remains `SOLVED_NEIGHBORHOOD_RADIUS=5`.
- Stream2 disabled-K2 fast path avoids calling suffix scan when `stream2_suffix.enabled==0`.
