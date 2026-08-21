# Molab IHES beam 2^22 parent-batch sweep — 2026-08-21

Hardware: one NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB reported VRAM).
Workload: IHES puzzle 1, output-1 batchnorm-folded FP16 MLP, requested/effective beam 4,194,304, touch radius 0, first-solution mode, depth limit 8.
Solver: `78f4612c9ebe68fc21b849e150542d298301a33c`.

`BEAM_B_MICRO` is a row budget for this output-1 MLP. The actual parent batch is row budget / 24 moves.

| Row budget | Parent batch | Full depth-6 seconds | Ring jobs | Stream3 jobs | Stream4 jobs | Peak sampled VRAM MiB | Result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 12,288 | 512 | 7.22447 | 6,151 | 1,538 | 300 | 2,121 | solved depth 8 |
| 18,432 | 768 | 7.69060 | 4,096 | 1,024 | 303 | 2,235 | solved depth 8 |
| 24,576 | 1,024 | 7.80013 | 3,073 | 769 | 298 | 2,361 | solved depth 8 |
| 30,720 | 1,280 | 7.97574 | 2,459 | 615 | 303 | 2,499 | solved depth 8 |
| 36,864 | 1,536 | 8.15151 | 2,048 | 512 | 296 | 2,633 | solved depth 8 |
| 43,008 | 1,792 | 8.44756 | 1,756 | 439 | 303 | 2,769 | solved depth 8 |
| 49,152 | 2,048 | 9.02126 | 1,537 | 385 | 300 | 2,907 | solved depth 8 |
| 73,728 | 3,072 | 9.30904 | 1,024 | 256 | 309 | 3,465 | solved depth 8 |

Current winner: parent batch 512 / row budget 12,288. This is 19.9% faster than the inherited 2,048-parent Kaggle seed at the steady full frontier. Every completed configuration reached 100% sampled utilization and showed no OOM, overflow, or fatal marker.

The foreground SSE driver emitted a heartbeat every ten seconds and prevented idle reclaim. The free Molab sandbox was nevertheless hard-terminated with HTTP 410 during the subsequent 896-parent run, so the lower batch boundary and beams above 2^22 require a fresh sandbox.
