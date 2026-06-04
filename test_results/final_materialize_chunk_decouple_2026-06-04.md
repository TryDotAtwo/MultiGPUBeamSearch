# Final Materialize Chunk Decoupling

Change: added `BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES` so final materialization buffer sizing can be capped independently from Stream3 batch size.

Expected effect: the large final request/response/history exchange buffers now scale with `min(local_frontier, BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES)` when the variable is nonzero, instead of always scaling with `STREAM3_BATCH_CANDIDATES`.

Verification status:
- Local WSL `bash -n` could not run in this Codex environment because Bash instance creation returned `E_ACCESSDENIED`.
- Docker `bash -n hpc/tune_8xa100_pipeline.sh hpc/mephi_8xa100_common.sh` passed in `gpu-dev:latest`.
- Docker CMake configure for `static_memory_cuda_tests production_runner` was blocked before compilation because `CUTLASS_DIR` is required and no `cutlass.h` was found under `/workspace` or `/opt` in the local image.
