# Kaggle v83 Static Hybrid History Tmp Arena Validation

- Date: 2026-05-27
- Kernel: `trydotatwo/cayley-beam-gpu-runner`, version 83
- Source commit: `e5bad9c`
- Hardware: Kaggle T4 x2
- Config: `BEAM_WIDTH=2**26`, `DEPTH_LIMIT=20`, `RUN_TIMEOUT_SEC=300`, `HISTORY_MODE=static_hybrid`, `HISTORY_SLOT_COUNT=2`, `HISTORY_DISK_PATH=/tmp/beam_history_arena`
- Output directory: `test_results/kaggle_v83_output/`
- Result row: `return_code=-200`, `seconds=301.0550402429999`

## Result

- The configured 300-second runner timeout fired; expected diagnostic outcome.
- Full output download succeeded because static history arena files were outside `/kaggle/working`.
- Log scan found no `cuda stream fatal`, `code=3002`, `terminate`, `what()`, `final selected count does not match score phase counts`, `threshold_end=0`, `arena exhausted`, or `disk write failed`.
- Both ranks completed depth 9 before timeout and kept `next_frontier_size=33554432` on full-beam depths.
- `history_bytes_stored_ram=0` through depth 9; disk-first storage remained active and RAM fallback was not needed.

## Rank 0 Tail

| depth | depth_sec | final_candidate_count | next_frontier_size | history_stored | history_ram | history_disk |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 2.6203 | 11748673 | 11748673 | 199191104 | 0 | 11212336 |
| 6 | 28.8349 | 33554432 | 33554432 | 736062016 | 0 | 199191104 |
| 7 | 64.2696 | 33554432 | 33554432 | 1272932928 | 0 | 736062016 |
| 8 | 66.4894 | 33554432 | 33554432 | 1809803840 | 0 | 1272932928 |
| 9 | 66.9104 | 33554432 | 33554432 | 2346674752 | 0 | 1809803840 |
