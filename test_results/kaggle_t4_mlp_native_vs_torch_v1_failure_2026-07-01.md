# Kaggle 2xT4 MLP Native vs Torch v1 Failure - 2026-07-01

Status: `KernelWorkerStatus.ERROR`.

Root cause: the native CUTLASS MLP benchmark completed on both T4 GPUs, but the Torch benchmark exited before timing because `tools/stream1_mlp_torch_benchmark.py` did not recognize the `initial_state_id` column used by `data/test.csv`.

Error evidence:

```text
ValueError: puzzle_id=0 not found in /tmp/beam_solver_mlp_benchmark/data/test.csv
```

Native evidence from downloaded Kaggle output:

- GPU0 best native point: `B_MICRO=2048`, `concurrency=4`, `candidates_per_sec=14564172.9`.
- GPU1 best native point: `B_MICRO=4096`, `concurrency=1`, `candidates_per_sec=15758828.6`.

Fix: accept `initial_state_id` as a valid puzzle/state id field in the Torch benchmark parser, then rerun Kaggle as v2.