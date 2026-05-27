# Depth Flow Trace Debug

date=2026-05-27
scope=debug-only
request=diagnose suspicious full/small frontier alternation without architecture changes

## Added diagnostics
- flag=`BEAM_DEBUG_DEPTH_FLOW_TRACE`
- kaggle_flag=`DEBUG_DEPTH_FLOW_TRACE`
- output_prefix=`depth_flow_trace`
- summary_fields=`frontier_size, generated, threshold_start, threshold_end, stream3_threshold_pass, stream3_unique, stream3_local_pending, stream3_remote_send, stream5_recv, stream3_local_write, stream3_local_spill, stream3_remote_write, stream3_remote_spill, stream4_clean_after_drain, stream4_dirty_after_drain, final_local_clean, final_global_less, final_global_equal, final_total_available, final_global_keep, final_threshold, final_candidate_count, final_request_count, next_frontier_size`
- ring_fields=`rank, depth, ring, generated, threshold_pass, unique_count, local_pending, remote_send`
- ring_log_policy=`first_two_rings, last_two_rings, any_ring_where_threshold_pass != generated`

## Intended diagnosis
- `generated >> stream3_threshold_pass` means threshold filtering starts before final selection and can explain frontier collapse.
- `stream3_threshold_pass >> stream3_unique` means Stream3 dedup/score-key path collapses candidates before Stream4.
- `stream3_unique != stream3_local_pending + stream3_remote_send` means owner split loses or double-counts candidates.
- `stream3_remote_send != peer stream5_recv` means Stream5 exchange loses or duplicates candidates.
- `stream3_local_write + stream3_remote_write + spills` vs `stream4_clean_after_drain` localizes losses between Stream3 writes and Stream4 drain.
- `stream4_clean_after_drain` vs `final_total_available` localizes losses inside final threshold/counting.

## Verification
- `git diff --check`: pass.
- `python -m json.tool kaggle/cayley-beam-gpu-runner.ipynb`: pass.
- `python -m json.tool kaggle/beam_kernel.ipynb`: pass.
- Docker clean debug build with `BEAM_DEBUG_DEPTH_FLOW_TRACE=ON`: `production_runner`, `static_memory_cuda_tests`, `dispatcher_cuda_tests` built.
- Docker clean debug tests with `BEAM_DEBUG_DEPTH_FLOW_TRACE=ON`: `static_memory_cuda_tests=pass`, `dispatcher_cuda_tests=pass`.
- Docker clean debug build with `BEAM_DEBUG_DEPTH_FLOW_TRACE=OFF`: `production_runner` and `static_memory_cuda_tests` built; `static_memory_cuda_tests=pass`.
- Docker clean debug test with `BEAM_DEBUG_DEPTH_FLOW_TRACE=OFF`: `dispatcher_cuda_tests` hit existing timing-sensitive scheduler invariant `stream4 conditional scheduler launched too many jobs`; no scheduler/test changes were made because approved scope was debug-only.

## Architecture impact
- scheduling_changed=false
- threshold_algorithm_changed=false
- final_selection_changed=false
- Stream4_dedup_algorithm_changed=false
- atomics_added=false
- runtime_effect_when_flag_off=compile-time disabled
