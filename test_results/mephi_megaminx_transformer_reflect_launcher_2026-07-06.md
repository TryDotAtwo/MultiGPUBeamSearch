# MEPhI Megaminx Transformer Reflect Launcher Update

Date: 2026-07-06

Scope:

- Let `hpc/solve_then_reflect.sh` run Megaminx original+reflected jobs with the explicit Stream1 transformer backend.
- Allow SLURM arrays to set puzzle id from `SLURM_ARRAY_TASK_ID` when `PUZZLE_ID` is not provided.
- Keep the same production runner path semantics as the single-run Megaminx transformer launcher.

Key behavior:

- `ORIGINAL_PUZZLE_ID=${PUZZLE_ID:-${SLURM_ARRAY_TASK_ID:-0}}`.
- `MEGAMINX_STREAM1_BACKEND=native_cuda_graph` selects the native CUTLASS/graph production runner.
- `BEAM_WEIGHT_DIR` defaults to `repo/weights/megaminx_vlad_transformer_fp16` when present.
- The script still clears history between original and reflected solves and restores `data/test.csv` on cleanup.

Verification:

```text
docker run ... bash -lc "bash -n hpc/solve_then_reflect.sh hpc/start_8xa100_libtorch_megaminx.sh hpc/mephi_8xa100_common.sh"
exit=0

git diff --check
exit=0
```