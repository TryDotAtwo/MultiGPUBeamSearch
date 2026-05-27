# Phase Lifetime Layout Verification 2026-05-27

## Change

- `layout_phase2_select` remains based at `scratch_pool + 0`.
- Dead Stream1/2/3/4/5 scratch buffers remain allowed to overlap phase-2 final buffers.
- Live phase-1 inputs required by final selection now start at or after `layout_phase2_select_bytes`.
- Protected live inputs: `survivor_shard`, `clean_count`, `shard_score_hist_a`, `shard_score_hist_b`, `shard_score_hist_active_index`, `threshold_hist_active_snapshot`, `local_score_hist`, `global_score_hist`, `current_threshold`, `threshold_initialized`, `current_threshold_active_index`, `threshold_request_local`, `threshold_request_global`.
- `layout_phase3_materialize` still overlays phase-2 temp buffers after the common selected-candidate prefix.

## Local Verification

- `static_memory_cuda_tests`: pass.
- `threshold_cuda_tests`: pass.
- `production_runner 0 8 1048576` in Docker: pass, `completed_depths=8`, `last_final_frontier_size=1048576`, `last_final_threshold=22230`, `puzzle_solved=0`.
- `dispatcher_cuda_tests`: failed pre-existing scheduler-count assertion `stream4 conditional scheduler launched too many jobs`; not changed in this patch because the current task is static scratch lifetime, not scheduler behavior.

## Memory Evidence

- 1xGPU smoke config reported `scratch_pool_bytes=401595648`, `layout_phase1_streams_bytes=401595648`, `layout_phase2_select_bytes=60900608`, `layout_phase3_materialize_bytes=184550656`.
- The corrected layout did not force `scratch_pool_bytes = phase1 + phase2 + phase3`.

## Next Validation

- Push source to GitHub `main`.
- Launch Kaggle T4x2 `BEAM_WIDTH=2**26`, `DEPTH_LIMIT=16`.
- Expected v76 failure signature should disappear if final selection no longer corrupts `survivor_shard`.
