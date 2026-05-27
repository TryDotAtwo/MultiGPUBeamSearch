# Kaggle v85 K1 Radius 4 States 1-10 Validation 2026-05-27

## Run
- kernel: `trydotatwo/cayley-beam-gpu-runner`
- version: `85`
- accelerator: `GPU T4 x2`
- source_commit: `824199a`
- config: `START_PUZZLE_ID=1`, `PUZZLE_COUNT=10`, `BEAM_WIDTH=2**26`, `DEPTH_LIMIT=20`, `RUN_TIMEOUT_SEC=300`
- K1 config: `SOLVED_NEIGHBORHOOD_RADIUS=4`, `SOLVED_NEIGHBORHOOD_MAX_ENTRIES=0`
- debug: `DEBUG_DEPTH_FLOW_TRACE=False`
- output_dir: `test_results/kaggle_v85_output/`

## Result
- solved_count: `10/10`
- return_code_all: `0`
- fatal_patterns: `0`
- timeout_patterns: `0`
- per_puzzle_seconds_sum: `24.77726`
- notebook_wall_seconds_approx: `155.63`

## Per Puzzle
| puzzle_id | solved | seconds | length | solution |
|---:|---:|---:|---:|---|
| 1 | 1 | 1.62875 | 1 | `BR` |
| 2 | 1 | 1.58286 | 2 | `BL.-L` |
| 3 | 1 | 1.55201 | 3 | `R.-DR.BR` |
| 4 | 1 | 1.62045 | 4 | `FR.-BL.DL.BR` |
| 5 | 1 | 1.63269 | 5 | `FR.DL.-B.-BR.U` |
| 6 | 1 | 2.10275 | 6 | `-FL.BL.-FL.-FR.-B.U` |
| 7 | 1 | 2.67419 | 7 | `-F.-BL.-BL.-R.-FR.-L.B` |
| 8 | 1 | 2.18162 | 6 | `F.DR.-FR.-DR.-L.-L` |
| 9 | 1 | 3.74714 | 9 | `-FR.-FR.-F.-R.FR.DR.F.L.B` |
| 10 | 1 | 6.0548 | 10 | `-B.-BL.-U.-BL.BR.FL.-DL.-R.-BL.F` |

## Notes
- Radius 4 reduced per-puzzle runtime sum from v84 radius 1 `298.7447s` to `24.77726s` on the same puzzle range.
- Deepest case `puzzle_id=10`: `depth_global_solved=5`, `solved_depth=6`, final `solution_length=10`.
- CPU solution validation passed for every emitted solution.
