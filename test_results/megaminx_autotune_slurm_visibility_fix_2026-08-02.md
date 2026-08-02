# Megaminx autotune SLURM visibility fix — 2026-08-02

## Cluster evidence

Job 33324 loaded the bundled NCCL successfully but all eight ranks failed in `ncclCommInitRank` with an unhandled CUDA error. Inspection showed autotune overwrote scheduler-provided `CUDA_VISIBLE_DEVICES`, unlike the normal solve entrypoint.

## Fix

The autotune job now requires and preserves SLURM-provided `CUDA_VISIBLE_DEVICES`. The requested `--gpus` list still determines the rank/GRES count, while actual device visibility remains scheduler-owned.

## Verification

Full portable suite: `188 passed, 3 skipped`.
