# Kaggle v80 depth-flow timeout diagnostic

- Date: 2026-05-27
- Kernel: `trydotatwo/cayley-beam-gpu-runner`, version 80
- Source commit before run: `e0e4596` (`Set Kaggle diagnostic timeout`)
- Hardware: Kaggle T4 x2
- Config focus: `BEAM_WIDTH=2**26`, `DEBUG_DEPTH_FLOW_TRACE=ON`, `RUN_TIMEOUT_SEC=300`
- Result: timeout path worked, `return_code=-200`, `seconds=301.250634206`
- Output: `test_results/kaggle_v80_output/`

## Key metrics

Rank 0 summary:

| depth | frontier | generated | stream3_threshold_pass | pass_pct | threshold_end | final_total_available | final_threshold | next_frontier |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 11743213 | 281837112 | 136032158 | 48.27% | 26545 | 67149083 | 26475 | 33554432 |
| 7 | 33554432 | 805306368 | 13369344 | 1.66% | 0 | 12591602 | 4294967295 | 6295801 |
| 8 | 6295801 | 151099224 | 98048344 | 64.89% | 26904 | 67148534 | 26745 | 33554432 |
| 9 | 33554432 | 805306368 | 13565952 | 1.68% | 0 | 12481052 | 4294967295 | 6240526 |
| 10 | 6240526 | 149772624 | 98846609 | 66.00% | 27066 | 67143454 | 26905 | 33554432 |
| 11 | 33554432 | 805306368 | 13565952 | 1.68% | 0 | 12436428 | 4294967295 | 6218214 |
| 12 | 6218214 | 149237136 | 98011920 | 65.68% | 27169 | 67124442 | 27020 | 33554432 |
| 13 | 33554432 | 805306368 | 13565952 | 1.68% | 0 | 12390369 | 4294967295 | 6195185 |
| 14 | 6195185 | 148684440 | 97956743 | 65.88% | 27232 | 67128023 | 27086 | 33554432 |

Rank 1 has the same pattern within small rank-balance differences.

## Diagnosis

- Alternation is not explained by natural dedup alone.
- Full-frontier depths publish `threshold_end=0` during the depth.
- `threshold_end=0` causes Stream3 threshold filtering to keep only about `1.66-1.68%` of generated candidates on full-frontier depths.
- Final selection then reports `final_total_available < GLOBAL_BEAM_WIDTH_EFFECTIVE` and `final_threshold=UINT32_MAX`, proving the periodic threshold was too aggressive.
- Code mapping found a likely cause in `cuda/dispatcher.cu`: the multi-rank final reset branch clears `clean_count`, `dirty_count`, `processing_flag`, spill counters, and threshold state, but does not clear `shard_score_hist_a`, `shard_score_hist_b`, or `shard_score_hist_active_index`.
- The single-GPU final reset branch does clear both shard histograms and the active histogram index.
- Consequence: the next multi-rank depth starts with stale per-shard Stream4 score histograms from the previous depth, while the current threshold state is reset to uninitialized. A later Stream5 periodic threshold update can allreduce stale histogram state and publish `threshold=0`, which explains the full-depth collapse.

## Scope

- No runtime architecture change was made.
- No C++ code change was made after this diagnosis.
- Recommended next fix, pending explicit approval: make the multi-rank final reset clear the same histogram and Stream3 write/ready service state that the single-GPU branch clears.
