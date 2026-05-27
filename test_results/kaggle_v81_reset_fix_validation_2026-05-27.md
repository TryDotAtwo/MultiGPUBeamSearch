# Kaggle v81 reset-fix validation

- Date: 2026-05-27
- Kernel: `trydotatwo/cayley-beam-gpu-runner`, version 81
- Source commit: `ab5d15b` (`Clear multi-rank reset histograms`)
- Hardware: Kaggle T4 x2
- Config: `BEAM_WIDTH=2**26`, `DEPTH_LIMIT=20`, `RUN_TIMEOUT_SEC=300`, `DEBUG_DEPTH_FLOW_TRACE=ON`
- Output directory: `test_results/kaggle_v81_output/`
- Result row: `return_code=-200`, `seconds=301.247405554`

## Result

- The configured 300-second runner timeout fired; this is expected for this diagnostic run.
- No `cuda stream fatal`, `code=3002`, `terminate`, `what()`, or `final selected count does not match score phase counts` patterns were found in rank logs.
- No `threshold_end=0` lines were found in rank logs.
- The previous full/small frontier collapse is gone.

## Key evidence

Rank 0:

| depth | frontier | generated | threshold_end | stream3_threshold_pass | pass_pct | final_threshold | next_frontier | depth_sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 11743181 | 281836344 | 26536 | 134793563 | 47.83% | 26494 | 33554432 | 29.5138 |
| 7 | 33554432 | 805306368 | 24240 | 187210850 | 23.25% | 24188 | 33554432 | 66.1654 |
| 8 | 33554432 | 805306368 | 23329 | 182699397 | 22.69% | 23277 | 33554432 | 64.7545 |
| 9 | 33554432 | 805306368 | 22790 | 179447476 | 22.28% | 22764 | 33554432 | 64.0632 |
| 10 | 33554432 | 805306368 | 22384 | 176610793 | 21.93% | 22334 | 33554432 | 63.3693 |

Rank 1 matched the same pattern with small rank-balance differences.

## Interpretation

- The reset cleanup fixed the stale-histogram `threshold=0` bug observed in Kaggle v80.
- From depth 7 onward the frontier remains full at `33554432` per rank instead of alternating between full and about `6.2M`.
- Runtime is now slower than v80 because the bug no longer drops most candidates on every other depth. Full-depth work remains real: `ring_slot_jobs=4096`, `stream3_jobs=4096`, `stream4_jobs=225-240`.
- Next performance work should be config tuning or pipeline optimization, not stale-threshold debugging.
