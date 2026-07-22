# RTX 3070 FF1 resource-policy gate (2026-07-22)

Configuration: native FP16 Stream1 transformer, `b_micro=512`, concurrency 1, CUDA Graph, selected `m128n128` policies for QKV/attention-out/FF2. Every candidate was compared in 20 alternating process pairs with a complete score dump.

## Results

| FF1 candidate | Baseline median | Candidate median | Relative result | Candidate wins | Exactness |
|---|---:|---:|---:|---:|---|
| stages 2 vs stages 3 | 8.4784 ms | 8.5494 ms | -0.83% | 6/20 | 40/40 identical SHA |
| warp 64x32 vs warp 64x64 | 8.47455 ms | 9.1253 ms | -7.13% | 0/20 | 40/40 identical SHA |

Reference SHA-256: `a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`.

## Nsight Compute 2025.1.1

| Policy | Duration | Registers/thread | Shared memory | Occupancy | No eligible | Math throttle |
|---|---:|---:|---:|---:|---:|---:|
| warp 64x64, stages 3 | 1.05 ms | 232 | 49.15 KiB | 16.39% | 74.79% | 2.62 |
| warp 64x64, stages 2 | 1.07 ms | 252 | 32.77 KiB | 16.39% | 75.13% | 2.74 |
| warp 64x32, stages 3 | 1.55 ms | 140 | block-limited | 16.64% | 79.89% | 4.80 |

Stage 2 saved shared memory but increased registers and did not raise occupancy. The 8-warp tile reduced registers but still exposed only eight active warps per SM and increased math-pipeline pressure. Production remains `m128n128 / warp 64x64 / stages 3`.
