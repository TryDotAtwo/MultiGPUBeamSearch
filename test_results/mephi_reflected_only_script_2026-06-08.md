# MEPhI Reflected-Only Script - 2026-06-08

Added `hpc/solve_reflected_only.sh`.

Purpose:

- Reuse an already found `ORIGINAL_SOLUTION`.
- Write the reflected synthetic puzzle id `9000000 + PUZZLE_ID`.
- Solve only the reflected puzzle.
- Print the inverted reflected candidate for the original puzzle.
- Avoid spending cluster time re-solving the original puzzle.

Cluster behavior:

- SLURM time limit: `24:00:00`.
- Same MEPhI common runtime, safe history cleanup, rank log behavior, and
  `BEAM_WEIGHT_DIR` override support as the existing launcher.
- Requires `ORIGINAL_SOLUTION` env var.

Verification:

- Docker `bash -n` passed for:
  - `hpc/solve_reflected_only.sh`
  - `hpc/mephi_8xa100_common.sh`

GPU runtime was not executed locally.
