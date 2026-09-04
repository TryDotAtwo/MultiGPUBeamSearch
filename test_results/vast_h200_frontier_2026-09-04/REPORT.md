# Vast 8xH200 Cube4 frontier tuning and solve gate — 2026-09-04/05

## Environment

- Instance: Vast `49878187`, 8x NVIDIA H200 143771 MiB.
- Live total rate: `$37.58223684210527/hour` (GPU plus disk).
- Solver revision: `3b756afe9e7636e53520fa6de7e60a167bbaa79a`.
- Stream1: native FP16 CUDA Graph, final-CLS attention, split QKV, fused input LayerNorm; Hopper experimental path disabled.
- Persistent shard capacity scale: `1000000` (1.0).

## Selected production profile

```text
requested_beam=2900000000
effective_beam=2900361216
world_size=8
local_beam=362545152
shard_count=64
shard_capacity_candidates=5664768
b_micro=192
stream1_concurrency=12
stream3_ring_slots=12
stream4_batch_candidates=524288
stream4_trigger_candidates=1048576
stream4_active_sort_slots=4
final_materialize_chunk_candidates=262144
final_exchange_scale_ppm=8000000
gpu_headroom_bytes=4294967296
```

Capacity evidence:

- 2.90B completed initialization, NCCL count exchange, and depth 0.
- 2.94B fit the static plan but failed at the first NCCL count exchange.
- 3.00B failed the static budget gate: `required=147513567824`, `budget=144821649408`.
- During the full 2.90B run, all GPUs used about 141820 MiB, leaving about 1951 MiB physically free.

## Pipeline sweep

Puzzle 10, requested beam 2.60B, exact search without solved-neighborhood shortcut, depth 6:

| Profile | Depth 6 seconds |
|---|---:|
| A100-shaped: micro 8192, concurrency/ring 1, shards 8, sort slots 1 | 46.8925 |
| H200 Stream1 only: micro 192, concurrency/ring 12, shards 8, sort slots 1 | 47.3152 |
| shards 16, sort slots 4, batch 524288 | 27.4171 |
| shards 32, sort slots 4, batch 524288 | 24.1334 |
| shards 64, sort slots 4, batch 524288 | 22.4457 |

The measured end-to-end improvement from the A100-shaped outer pipeline to the selected 64-shard profile was about 2.09x. Isolated Stream1 throughput was not used as an end-to-end claim.

## Correctness gate

Puzzles 1 through 10 were each solved at the selected 2.90B profile. Every original and reflected run completed, and every final verification printed both `original_solution_solves_original=1` and `candidate_solution_solves_original=1`.

Original solution lengths were: `2, 3, 4, 5, 6, 7, 6, 7, 10, 9`.

## Puzzle 1000 partial run

Puzzle 1000 was launched after the 1-10 gate and stopped on explicit user request during depth 9. It was not solved and depth 10 was not reached.

Completed timings:

- depth 6: `22.7391 s`, next local frontier `165361436`;
- depth 7: `434.35 s`, next local frontier `362545152` (global beam saturated);
- depth 8: `931.266 s`, next local frontier `362545152`;
- depth 9: interrupted, no completed timing.

At the live rate, one completed saturated depth cost `931.266 / 3600 * 37.58223684210527 = $9.72`. A complete puzzle cost cannot be reported because puzzle 1000 was intentionally stopped before a solution. The logs preserve the exact partial-run evidence.

## Artifact policy

The directory contains all downloaded ordinary logs, per-rank logs, GPU monitor logs, capacity failures, calibration runs, gate runs, and the interrupted puzzle-1000 run. The original compressed bundle is retained as `h200_frontier_2026-09-04.tar.gz`.

Large transient history arenas were deliberately excluded: they occupied about 149 GiB remotely, are reconstructable runtime scratch/history, and are unsuitable for GitHub. No completed solution proof depends on those excluded files.
