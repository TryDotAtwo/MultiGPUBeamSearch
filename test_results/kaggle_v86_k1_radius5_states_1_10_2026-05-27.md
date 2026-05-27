# Kaggle v86 K1 Radius 5 States 1-10 Validation 2026-05-27

## Run
- kernel: `trydotatwo/cayley-beam-gpu-runner`
- version: `86`
- accelerator: `GPU T4 x2`
- source_commit: `4e3d8be`
- config: `START_PUZZLE_ID=1`, `PUZZLE_COUNT=10`, `BEAM_WIDTH=2**26`, `DEPTH_LIMIT=20`, `RUN_TIMEOUT_SEC=300`
- K1 config: `SOLVED_NEIGHBORHOOD_RADIUS=5`, `SOLVED_NEIGHBORHOOD_MAX_ENTRIES=0`
- debug: `DEBUG_DEPTH_FLOW_TRACE=False`
- output_dir: `test_results/kaggle_v86_output/`

## Result
- solved_count: `10/10`
- return_code_all: `0`
- fatal_patterns: `0`
- timeout_patterns: `0`
- per_puzzle_seconds_sum: `20.87583`
- notebook_wall_seconds_approx: `178.93`

## Per Puzzle
| puzzle_id | solved | seconds | length | solution |
|---:|---:|---:|---:|---|
| 1 | 1 | 1.66516 | 1 | `BR` |
| 2 | 1 | 1.59812 | 2 | `BL.-L` |
| 3 | 1 | 1.58386 | 3 | `R.-DR.BR` |
| 4 | 1 | 1.7064 | 4 | `FR.-BL.DL.BR` |
| 5 | 1 | 1.70154 | 5 | `FR.DL.-B.-BR.U` |
| 6 | 1 | 1.63304 | 6 | `BL.-FL.-FL.-FR.-B.U` |
| 7 | 1 | 2.27959 | 7 | `-F.-BL.-R.-FR.-BL.-L.B` |
| 8 | 1 | 1.7332 | 6 | `F.DR.-FR.-DR.-L.-L` |
| 9 | 1 | 3.19066 | 9 | `-FR.-FR.-F.-R.FR.DR.F.L.B` |
| 10 | 1 | 3.78426 | 10 | `-B.-BL.-U.-BL.BR.FL.-DL.-R.-BL.F` |

## Comparison
- Radius 1 per-puzzle seconds sum: `298.7447`
- Radius 4 per-puzzle seconds sum: `24.77726`
- Radius 5 per-puzzle seconds sum: `20.87583`
- Radius 5 is faster inside `production_runner` than radius 4 on this 10-puzzle set, but notebook wall time is higher because radius 5 precompute/setup cost is larger.

## Notes
- Deepest case `puzzle_id=10`: `depth_global_solved=4`, `solved_depth=5`, final `solution_length=10`.
- CPU solution validation passed for every emitted solution.
