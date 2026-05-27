# Kaggle v82 Static Hybrid History Validation

- Date: 2026-05-27
- Kernel: `trydotatwo/cayley-beam-gpu-runner`, version 82
- Source commit: `2636e41`
- Hardware: Kaggle T4 x2
- Config: `BEAM_WIDTH=2**26`, `DEPTH_LIMIT=20`, `RUN_TIMEOUT_SEC=300`, `HISTORY_MODE=static_hybrid`, `HISTORY_SLOT_COUNT=2`
- Output directory: `test_results/kaggle_v82_output/`
- Result row: `return_code=-200`, `seconds=300.84553840399997`

## Result

- The configured 300-second runner timeout fired; this was expected for the diagnostic run.
- Log scan found no `cuda stream fatal`, `code=3002`, `terminate`, `what()`, `final selected count does not match score phase counts`, `threshold_end=0`, `arena exhausted`, or `disk write failed`.
- Rank 0 and rank 1 both completed depth 10 before timeout.
- `history_bytes_stored_ram=0` through depth 10, so all completed history writes stayed on disk arena.
- `history_bytes_stored_disk` was present in depth logs, confirming the new `static_hybrid` path was active.

## Key Rank 0 Lines

| depth | depth_sec | final_candidate_count | next_frontier_size | history_stored | history_ram | history_disk |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 29.8532 | 33554432 | 33554432 | 735969216 | 0 | 199098304 |
| 7 | 66.1048 | 33554432 | 33554432 | 1272840128 | 0 | 735969216 |
| 8 | 64.9948 | 33554432 | 33554432 | 1809711040 | 0 | 1272840128 |
| 9 | 65.0356 | 33554432 | 33554432 | 2346581952 | 0 | 1809711040 |
| 10 | 64.9540 | 33554432 | 33554432 | 2883452864 | 0 | 2346581952 |

## Output Packaging Note

- Full `kaggle kernels output` initially timed out because the first config placed sparse static history arena files under `/kaggle/working/history_arena`.
- A filtered download for CSV/log files succeeded.
- The Kaggle config was changed to `/tmp/beam_history_arena` after v82 so future outputs do not package static history arena files.
