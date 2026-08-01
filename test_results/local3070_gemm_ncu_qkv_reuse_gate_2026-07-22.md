# RTX 3070 current GEMM profile and QKV reuse gate (2026-07-22)

Nsight Compute 2025.1.1 profiled the selected `m128n128 / warp 64x64 / stages 3` kernels at `b_micro=512`, concurrency 1.

| Family | Duration | DRAM throughput | SM throughput | Registers/thread | Occupancy | No eligible | Math throttle |
|---|---:|---:|---:|---:|---:|---:|---:|
| QKV | 695.58 us | 44.50% | 46.17% | 232 | 16.32% | 87.26% | 10.26 |
| attention-out | 254.62 us | 53.57% | 42.04% | 228 | 15.80% | 86.08% | 7.42 |
| FF2 | 954.40 us | 37.96% | 44.91% | 228 | 15.86% | 90.72% | 15.90 |

A QKV `threadblock 256x128 / warp 64x64` candidate doubled B-tile reuse while preserving the warp accumulation shape. Across 20 alternating pairs it was exact in all 40 dumps but regressed from median 9.0855 ms to 9.33925 ms (-2.72%) and won 4/20 pairs. One candidate process was a 41 ms system outlier and was not needed for the median decision.

NCU confirmed that relative DRAM pressure fell 44.50% to 36.43%, but the larger block reduced the resident-block limit from two to one, worsened No Eligible 87.26% to 89.88% and math throttle 10.26 to 13.49, and increased QKV duration 695.58 us to 849.09 us. Production remains `m128n128`.

All complete dumps used SHA-256 `a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`.
