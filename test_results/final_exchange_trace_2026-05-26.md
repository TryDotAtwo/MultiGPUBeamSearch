# Final Exchange Trace Diagnostic

Date: 2026-05-26

## Context

- Target issue: Kaggle 2xT4 `WORLD_SIZE=2` failure at depth 1 on rank 1.
- Failure line: `final request return rank exceeds WORLD_SIZE`.
- Previous fix status: send-count/send-offset snapshot in `exchange_u64_items` did not remove the failure in Kaggle version 50.
- User-approved scope: diagnostics only; no final-exchange algorithm change.

## Code Change

- Added CMake option `BEAM_DEBUG_FINAL_EXCHANGE_TRACE`.
- Effective compile gate: `BEAM_ENABLE_DEBUG=ON` and `BEAM_DEBUG_FINAL_EXCHANGE_TRACE=ON`.
- Production effect when disabled: trace code is excluded by preprocessor guard.
- Trace coverage:
  - global final threshold summary;
  - less/equal counts and offsets by rank;
  - selected candidate samples with route fields;
  - history/request/response exchange send and receive counts;
  - first 8 and last 4 `FinalRequest` records for send/receive/response paths;
  - invalid request payload before the existing fatal throw.

## Local Verification

- `git diff --check`: passed.
- Kaggle notebook JSON parse: passed.
- Docker CUDA targeted tests:
  - `stream5_cuda_tests`: passed.
  - `final_cuda_tests`: passed.
  - `threshold_cuda_tests`: passed.
  - `dispatcher_cuda_tests`: passed.
  - `static_memory_cuda_tests`: passed.
- `WORLD_SIZE=1` smoke:
  - command: `BEAM_GPU_HEADROOM_BYTES=0 BEAM_HISTORY_MODE=ram ./build-docker/production_runner 0 1 4194304`.
  - result: `puzzle_solved=0 puzzle_id=0 seconds=0.358157 solution_length=-1 solution=`.

## Pending

- Push diagnostic patch to GitHub `main`.
- Launch Kaggle 2xT4 with `BEAM_DEBUG_FINAL_EXCHANGE_TRACE=ON`.
- Inspect `final_exchange_trace*` lines around the first invalid `FinalRequest`.
