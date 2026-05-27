# Kaggle v77 Beam 2**26 Depth 16 Validation 2026-05-27

## Run

- Kernel: `trydotatwo/cayley-beam-gpu-runner`, version `77`.
- Source commit: `d037caf Fix final select live input overlay`.
- Hardware: Kaggle `NvidiaTeslaT4`, `WORLD_SIZE=2`.
- Config: `BEAM_WIDTH=2**26`, `DEPTH_LIMIT=16`, `RUN_TIMEOUT_SEC=600`.
- Output directory: `test_results/kaggle_v77_output/`.

## Result

- `beam_run_results.csv`: `return_code=0`.
- Rank 0: `completed_depths=16`, `elapsed_sec=541.086`, `last_final_frontier_size=33554432`, `last_final_threshold=26304`, `threshold_updates=146`.
- Rank 1: `completed_depths=16`, `elapsed_sec=541.081`, `last_final_frontier_size=33554432`, `last_final_threshold=26304`, `threshold_updates=146`.
- Puzzle status: `puzzle_solved=0`, no solution by depth 16.

## Static Memory

- `gpu_budget_bytes=14640021504`.
- `static_allocation_bytes=12300679168`.
- `estimated_required_device_bytes=12391290400`.
- `scratch_pool_bytes=8005674752`.
- `layout_phase1_streams_bytes=7338354432`.
- `layout_phase2_select_bytes=2485650688`.
- `layout_phase3_materialize_bytes=8005674752`.

## Depth 15 Evidence

- Rank 0: `depth_done=15`, `depth_sec=20.0084`, `stream4_jobs=87`, `stream4_busy_max=2`, `global_spill_peak=0`, `final_candidate_count=33554432`, `next_frontier_size=33554432`.
- Rank 1: `depth_done=15`, `depth_sec=20.0086`, `stream4_jobs=87`, `stream4_busy_max=2`, `global_spill_peak=0`, `final_candidate_count=33554432`, `next_frontier_size=33554432`.

## Failure Pattern Scan

- No `terminate`.
- No `cuda stream fatal`.
- No `final selected count does not match score phase counts`.
- No `does not match`.
- No `exact_invariant_ok=0`.

## Interpretation

- The v76 final-selection corruption signature disappeared after moving phase-1 live final-selection inputs outside the phase-2 select overlay range.
- `scratch_pool_bytes` remained `max(phase1, phase2, phase3)`, not `phase1 + phase2 + phase3`.
- Kaggle status ended as `CANCEL_ACKNOWLEDGED`, but downloaded rank logs and `beam_run_results.csv` show both rank processes completed normally with `return_code=0`.
