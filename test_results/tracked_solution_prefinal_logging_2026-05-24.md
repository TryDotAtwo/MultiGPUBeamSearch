# Tracked Solution Prefinal Logging 2026-05-24

- goal: distinguish three loss modes for known solution prefixes: absent before final cut, present before final cut but pruned by score threshold, or present after score threshold but lost by final cut/load-balance.
- added_log: `track_solution_prefinal` emitted before `final_filter_load_balance_cuda`, after final-local Stream4 dedup/flush and final threshold computation.
- prefinal_fields: `found`, `matches`, `first_index`, `best_index`, `best_score_key`, `final_threshold`, `threshold_pass`, `threshold_margin`, `parent_idx`, `route_packed`, `source_rank`, `owner`, `move`, `move_name`, `shard`, `local`, `final_candidate_count`.
- expanded_log: `track_solution_prefix` now emits the same score/route fields for the final candidate buffer after final filtering.
- implementation: GPU block scan over `survivor_shard` with no atomics; block summaries are copied to CPU only when `BEAM_TRACK_SOLUTION_PATH` is enabled.
- local_verification: `git diff --check` passed.
- kaggle_v20_result: compile failed because host code called `__device__ invalid_track_candidate_device()` in `scan_tracked_prefinal_hash`.
- fix_after_v20: host code now uses a host-side invalid `CandidateMeta` literal; resubmit required.
- local_blocker: Docker Desktop remains unavailable, so local CUDA build/tests were not run before Kaggle push.
