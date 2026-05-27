# Static Final Phase Layout 2026-05-27

## Change
- Implemented explicit static scratch layouts:
  - `layout_phase1_streams_bytes`
  - `layout_phase2_select_bytes`
  - `layout_phase3_materialize_bytes`
- Updated scratch sizing:
  - `scratch_pool_bytes = max(layout_phase1_streams_bytes, layout_phase2_select_bytes, layout_phase3_materialize_bytes)`
- Kept persistent buffers outside scratch:
  - `current_frontier_states`
  - `solved_*`
- Kept final cross-phase outputs in a common scratch prefix:
  - multi-rank `final_selected_buffer`
  - single-rank `final_candidate_buffer`
  - single-rank `final_request_buffer`
  - shared counts, validation, send/recv service buffers
- Overlaid final temp regions after common prefix:
  - phase2 select temp: `final_keep_flags`, `final_block_counts`, `final_block_offsets`
  - phase3 materialize temp: `final_candidate_buffer` for multi-rank, `next_frontier_states_tmp`, request/response/history slots, CUB temp

## Verification
- `git diff --check`: pass
- Docker CUDA build target: `static_memory_cuda_tests production_runner`: pass
- Docker CUDA test: `./build-docker/static_memory_cuda_tests`: pass
- Docker GPU smoke:
  - command: `BEAM_RUNTIME_CONFIG_MODE=manual BEAM_STREAM3_RING_SLOTS=1 BEAM_SHARD_COUNT=4 BEAM_SHARD_BUFFER_COUNT=2 BEAM_STREAM4_BATCH_CANDIDATES=262144 BEAM_STREAM4_TRIGGER_CANDIDATES=262144 BEAM_SHARD_CAPACITY_CANDIDATES=327680 BEAM_GLOBAL_SPILL_CAPACITY=0 ./build-docker/production_runner 0 8 1048576`
  - result: pass
  - output after forced A/B config contract: `puzzle_solved=0 puzzle_id=0 seconds=27.469 solution_length=-1`

## Notes
- Windows local CMake build did not compile because local Windows CMake could not find a CUDA toolset.
- Docker CUDA build/test was used as the verification environment.
- `SHARD_BUFFER_COUNT=2` is now enforced by config. Every logical shard always has resident A/B physical shard buffers.
- Legacy `global_spill_capacity` is only for unreachable single-buffer Stream4 mode. Target A/B mode uses `GLOBAL_SPILL_CAPACITY=0`.
