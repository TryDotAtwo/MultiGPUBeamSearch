# Multi-rank final reset cleanup fix

- Date: 2026-05-27
- Context: Kaggle v80 depth-flow diagnostics showed full-depth collapse after multi-rank periodic threshold updates published `threshold=0`.
- Root cause candidate: multi-rank final reset did not clear per-depth Stream4 shard histograms, unlike the single-GPU reset path.
- Change: `cuda/dispatcher.cu` multi-rank final reset now clears:
  - `stream3_ready_flag`
  - `stream3_ready_shard_list`
  - `stream3_ready_count`
  - `stream3_write_buffer_index`
  - `shard_score_hist_a`
  - `shard_score_hist_b`
  - `shard_score_hist_active_index`
- Architecture impact: none. This is per-depth state cleanup aligned with the already-existing single-GPU reset.
- Expected result: next depth cannot build a threshold from stale previous-depth Stream4 histograms.

## Verification

- `git diff --check`: pass
- Docker build: `production_runner dispatcher_cuda_tests static_memory_cuda_tests`: pass
- Docker targeted tests: `dispatcher_cuda_tests`, `static_memory_cuda_tests`: pass
