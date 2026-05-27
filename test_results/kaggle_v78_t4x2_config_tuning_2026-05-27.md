# Kaggle v78 T4x2 Config Tuning 2026-05-27

## Run

- Kernel: `trydotatwo/cayley-beam-gpu-runner`, version `78`.
- Hardware: Kaggle T4 x2, `WORLD_SIZE=2`.
- Source: GitHub `main`.
- Config intent: config-only speed tuning, no architecture/code changes.
- Output directory: `test_results/kaggle_v78_tuning_output/`.

## Config Delta

- `DEPTH_LIMIT=20`, `RUN_TIMEOUT_SEC=900`.
- Heavy debug traces disabled: `ENABLE_DEBUG_LOGS=False`, all `DEBUG_*` trace flags false.
- Depth logs preserved: `ENABLE_DEBUG=True`, `ENABLE_DEPTH_LOGS=True`, `DEPTH_LOG_EVERY=1`.
- Live log noise reduced: `LIVE_LOG_RANKS=[0]`.
- Stream4 tuning: `SHARD_COUNT=8`, `SHARD_CAPACITY_SCALE_PPM=1125000`, `STREAM4_ACTIVE_SORT_SLOTS=4`.
- Kept: `STREAM4_BATCH_CANDIDATES=196608`, `STREAM4_TRIGGER_CANDIDATES=786432`, `SHARD_BUFFER_COUNT=2`, `GLOBAL_SPILL_CAPACITY=0`.

## Result

- `beam_run_results.csv`: `return_code=0`.
- Rank seconds: rank0 `492.638`, rank1 `492.644`.
- Completed depths: `0..19`, `DEPTH_LIMIT=20`.
- No `terminate`, `cuda stream fatal`, `does not match`, `exact_invariant_ok=0`, or `exceeds GPU budget` matches.
- Puzzle status: `puzzle_solved=0`, no solution by depth 20.

## Runtime Summary

- v78 rank0 depth sum `0..15`: `339.230s`.
- v77 rank0 depth sum `0..15`: `539.504s`.
- Delta versus v77 diagnostic run: `-200.274s`, about `37.1%` lower wall time through depth 15.
- v78 rank0 depth sum `0..19`: `491.021s`.
- v78 full-input depths with `ring_slot_jobs=4096`: average `49.370s`.
- v78 small-input depths after depth 7 with `ring_slot_jobs!=4096`: average `18.583s`.

## Key Depths

| depth | depth_sec | ring_slot_jobs | stream4_jobs | threshold_updates | final_threshold | next_frontier_size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 28.5245 | 1434 | 176 | 11 | 26466 | 33554432 |
| 7 | 43.1353 | 4096 | 24 | 2 | 4294967295 | 6292611 |
| 8 | 18.5590 | 769 | 126 | 8 | 26738 | 33554432 |
| 9 | 47.2560 | 4096 | 24 | 2 | 4294967295 | 6233509 |
| 14 | 17.8922 | 752 | 126 | 8 | 27106 | 33554432 |
| 15 | 47.0673 | 4096 | 24 | 2 | 4294967295 | 6185123 |
| 18 | 18.8224 | 754 | 126 | 8 | 27147 | 33554432 |
| 19 | 67.1988 | 4096 | 232 | 15 | 24791 | 33554432 |

## Interpretation

- Config-only tuning succeeded and improved the depth-0..15 benchmark substantially compared with the previous heavy-diagnostic v77 run.
- The full/small frontier alternation remains. Config tuning reduces constant factors but cannot remove the algorithmic alternation caused by `current_frontier_size`.
- Depth 19 is slower because it combines full input (`4096` ring jobs) with full output and many Stream4 jobs (`232`), unlike the `UINT32_MAX` threshold depths that shrink output to about `6.1M`.
- Depth-only logs are not the dominant runtime factor in this run; heavy trace flags were compiled off, and output logs are small.

