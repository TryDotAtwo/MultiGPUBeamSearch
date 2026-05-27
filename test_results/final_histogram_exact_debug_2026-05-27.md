# Final Histogram Exact Debug

## Scope
- Date: 2026-05-27
- Change type: diagnostics only
- Compile flag: `BEAM_DEBUG_FINAL_HISTOGRAM_TRACE`
- Runtime architecture changes: none
- New CUDA kernels: none
- New atomics: none
- New static buffers: none

## Added Diagnostics
- Host exact histogram from `survivor_shard + clean_count`, copied one physical shard at a time.
- Per-shard comparison between active Stream4 histogram and exact host scan.
- Aggregate local exact histogram comparison against `local_score_hist`.
- Threshold-window bin dump for `[final_threshold-8, final_threshold+8]`.
- Top 32 local score-bin differences by absolute count.

## Purpose
- Localize Kaggle v75 failure class:
  - `score_key_mismatch`
  - `histogram_snapshot_stale`
  - `threshold_prefix_bug`

## Expected Signal
- If exact host histogram matches exact GPU counts but differs from Stream4 histogram, inspect Stream4 histogram population/active-index commit.
- If exact host histogram matches Stream4 histogram but final threshold selection still fails, inspect threshold prefix/selection.
- If only selected shards differ, inspect final drain and A/B histogram snapshot timing.

## Local Verification
- `git diff --check`: pass
- Kaggle notebook JSON parse: pass
- Docker build target `production_runner dispatcher_cuda_tests threshold_cuda_tests static_memory_cuda_tests`: pass
- Docker `threshold_cuda_tests`: pass
- Docker `dispatcher_cuda_tests`: pass
- Docker `static_memory_cuda_tests`: pass
