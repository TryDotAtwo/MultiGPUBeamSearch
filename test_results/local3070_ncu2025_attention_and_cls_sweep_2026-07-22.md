# RTX 3070 NCU recovery and final-CLS attention sweep — 2026-07-22

## Nsight Compute recovery

The image contains NCU 2025.1.1 and 2026.1.1. `ncu` in PATH selected 2026.1.1, which failed against driver 572.70 with CUDA error 200 while loading internal profiler modules. Explicit `/opt/nvidia/nsight-compute/2025.1.1/ncu` works; both basic and 40-pass full profiles completed.

Full q64k64 FMHA counters at b512/c1: duration 370.34 us, memory throughput 47.64%, SM throughput 58.58%, tensor-pipe active 27.27%, 125 registers/thread, 18.94 KiB dynamic shared/block, 33.33% theoretical and 32.32% achieved occupancy. Registers limit the SM to four blocks. Shared stores have 188,347 bank conflicts across 688,128 requests; NCU estimates 5.249% opportunity. Schedulers have only 1.15 eligible warps from 3.88 active and no eligible warp in 40.77% of cycles.

## Final-CLS exact sweeps

Every b512/c1 dump across full-final, Q=1 q64k64, and Q=1 q32k64 was byte-identical (60/60): `a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`.

| c1 variant | Median, ms | Decision |
|---|---:|---|
| full-final q64k64 | 8.9312 | pipeline reference |
| Q=1 q64k64 | 8.4802 | isolated +5.05%, pipeline rejected |
| Q=1 q32k64 | 8.4895 | no gain over Q=1 q64 |

At concurrency 2, all 40 dumps were identical (`ab4ceab8...91424`), but Q=1 regressed from 18.8013 to 19.2344 ms (2.30%).

Six alternating c1 Stream1-to-2-to-3 pairs produced medians 83.8032 ms for full-final and 84.8628 ms for Q=1, a 1.26% regression. Therefore the exact-signature cache retains full-final attention.

A scalar one-warp Q=1 experiment changed the complete dump and was removed before final selection. Only exact tensor-core policies remain compiled.

Artifacts: `local3070_ncu2025_selected_fmha_full_2026-07-19.ncu-rep` and `.txt`.