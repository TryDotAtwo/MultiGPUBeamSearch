# Molab IHES beam 2^22 parent-batch sweep — 2026-08-21

> Superseded for profile selection: the user subsequently restricted Molab
> tuning to `output_dim=24` only. None of the output-1 measurements in this
> report may be used to select or publish a Molab runtime profile.

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

Additional lower-bound rerun in a fresh sandbox:

| Row budget | Parent batch | Full depth-6 seconds | Ring jobs | Stream3 jobs | Stream4 jobs | Peak sampled VRAM MiB | Result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 6,144 | 256 | 7.62404 | 12,301 | 3,076 | 296 | 2,053 | solved depth 8 |
| 9,216 | 384 | 7.39497 | 8,192 | 2,048 | 297 | 2,073 | solved depth 8 |
| 12,288 | 512 | 7.23221 | 6,151 | 1,538 | 300 | 2,121 | solved depth 8 |

Current winner: parent batch 512 / row budget 12,288. This is 19.9% faster than the inherited 2,048-parent Kaggle seed at the steady full frontier. Every completed configuration reached 100% sampled utilization and showed no OOM, overflow, or fatal marker.

The foreground SSE driver emitted a heartbeat every ten seconds and prevented idle reclaim. The free Molab sandbox was nevertheless hard-terminated with HTTP 410 during the subsequent 896-parent run, so the lower batch boundary and beams above 2^22 require a fresh sandbox.

The 512-parent optimum was then carried to beam `2**23`. Full-frontier depth-6 results were 18.5218 s at parent 384, 18.2789 s at parent 512, and 18.2024 s at parent 768. All three solved at depth 8, used about 4,059 MiB sampled VRAM, and showed no OOM/overflow. The 512-to-768 difference is only 0.42%, so 512 is the current conservative profile choice. Molab terminated the entire sandbox before the queued `2**24` cell could start, despite the workload running as a normal visible marimo cell; this rules out detached-process and scratchpad-idle reclaim as the sole cause.

## Corrected isolated Stream 1 matrix

The IHES contract is `MOVE_COUNT=18`, not 24. A fresh native `stream_benchmark`
run used row budget `parent_batch * 18`, FP16 output-1 batchnorm-folded MLP, and
concurrency `{1,2,4,8,16,32}` on the same RTX PRO 6000. It completed 130 of
144 requested points before Molab terminated the sandbox and pair endpoint.
No solver profile was changed from this partial evidence.

| Parent batch | Concurrency | Throughput parents/s | Candidates/s | Scratch bytes |
|---:|---:|---:|---:|---:|
| 320 | 32 | 731,741.6 | 13,171,348.9 | 1,109,237,760 |
| 448 | 8 | 726,765.4 | 13,081,777.7 | 388,233,216 |
| 256 | 32 | 723,931.4 | 13,030,765.0 | 887,390,208 |
| 320 | 16 | 719,677.9 | 12,954,202.6 | 554,618,880 |
| 640 | 4 | 715,251.1 | 12,874,519.9 | 277,309,440 |

Observed large-batch tail was already declining: parent 12,288 reached
499,668.5 parents/s at concurrency 1 and about 446k parents/s at concurrency
16-32; parent 16,384 reached 437,263.7 at concurrency 1 and 421,182.5 at
concurrency 8. The missing 14 points are parent 16,384 at concurrency 16/32
and all concurrency values for parent 24,576 and 32,768. A short tail-only run
is still required for a complete matrix, followed by a fine sweep around
parent batches 256-640.
