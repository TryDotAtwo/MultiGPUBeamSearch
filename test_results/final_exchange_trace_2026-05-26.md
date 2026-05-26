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

- Push offset-plan/alignment fix to GitHub `main`.
- Launch Kaggle version 52 on GPU T4 x2.

## Kaggle Version 51 Result

- Kaggle version: 51.
- Hardware: GPU T4 x2.
- Status: rank 1 threw at depth 1; rank 0 then waited in the next NCCL phase, so the notebook looked hung.
- Fatal line: `rank=1 what(): final request return rank exceeds WORLD_SIZE`.
- Direct trace evidence:
  - rank 1 depth 1 `request_send_counts`: `count0=97 count1=112 offset0=0 offset1=97 offset2=209`.
  - rank 1 depth 1 `history_recv`: `count0=0 count1=204 offset0=0 offset1=0 offset2=204`.
  - rank 1 depth 1 `request_send_snapshot`: `count0=97 count1=112 offset0=0 offset1=0 offset2=204`.
- Diagnosis: `request` send offsets were stored in `recv_offsets`; the preceding `history` exchange reused and overwrote `recv_offsets` before `request` exchange used it.
- Corruption sample: rank 1 `request_recv index=0 parent_idx=6615865701556808792 target_local_idx=563486153 return_rank=20906 move=98 pad=213`.

## Fix

- Added `FinalExchangePlan{count, offset, total}`.
- `history_plan`, `request_plan`, and `response_plan` are independent immutable host-side send plans.
- `exchange_u64_items` now snapshots the send plan and creates a local receive plan from received counts.
- Removed shared host scratch offset reuse across `history`, `request`, and `response`.
- Changed `FinalHistoryRecord` to `alignas(32)` and `sizeof(FinalHistoryRecord)==64`.
- Aligned `next_frontier_states_tmp` and `final_response_buffer` to at least `alignof(CandidateMeta)` because multi-rank final overlays these buffers with `CandidateMeta` and `FinalHistoryRecord`.
- Aligned final count/offset device service buffers to 256 bytes.

## Fix Verification

- Docker build target `production_runner`: passed.
- Docker targeted CUDA tests:
  - `stream5_cuda_tests`: passed.
  - `final_cuda_tests`: passed.
  - `threshold_cuda_tests`: passed.
  - `dispatcher_cuda_tests`: passed.
  - `static_memory_cuda_tests`: passed.
- `WORLD_SIZE=1` smoke:
  - command: `BEAM_GPU_HEADROOM_BYTES=0 BEAM_HISTORY_MODE=ram ./build-docker/production_runner 0 1 4194304`.
  - result: `puzzle_solved=0 puzzle_id=0 seconds=0.304287 solution_length=-1 solution=`.
