# RTX 3070 QKV/FF1 stages=4 gate — 2026-07-22

## Decision

Reject and remove the temporary CUTLASS `stages=4` policy for QKV and FF1. Production remains QKV/FF1 `m128n128`, identity swizzle 8, stages 3.

## Measurement validity

The earlier timing series in queue log `.gpu_queue/logs/427a3fb879b7.log` is excluded because the user reported possible overlap with an interactive game. It is not used below.

The accepted rerun was bookended by idle checks. Before the A/B run, five consecutive `nvidia-smi dmon` samples reported `SM=0%` and `MEM=0%`, and the compute-process query returned no rows. A later idle check before profiling again reported five consecutive zero-load samples.

Evidence logs:

- pre-run utilization: `.gpu_queue/logs/7671751cd830.log`
- pre-run compute processes: `.gpu_queue/logs/1407995dab9d.log`
- clean alternating A/B: `.gpu_queue/logs/b921b0cacfbb.log`
- pre-profile utilization: `.gpu_queue/logs/927849e20f16.log`

## Exactness and speed

All 48 complete score dumps matched canonical SHA-256:

`a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`

| Configuration | Median, ms | Change vs stages3 | Paired wins |
|---|---:|---:|---:|
| QKV3 + FF1-3 | 7.79120 | baseline | — |
| QKV4 + FF1-3 | 8.18075 | +5.00% | 0/12 |
| QKV3 + FF1-4 | 9.33615 | +19.83% | 0/12 |
| QKV4 + FF1-4 | 9.76200 | +25.30% | 0/12 |

## Nsight Compute 2025.1.1 explanation

| Kernel | Stages | Duration | Dynamic shared/CTA | Theoretical occupancy | No eligible warp |
|---|---:|---:|---:|---:|---:|
| QKV | 3 | 693.41 us | 49.15 KiB | 16.67% | 87.20% |
| QKV | 4 | 852.32 us | 65.54 KiB | 8.33% | 89.26% |
| FF1 | 3 | 1.04 ms | 49.15 KiB | 16.67% | 74.63% |
| FF1 | 4 | 1.90 ms | 65.54 KiB | 8.33% | 85.90% |

The fourth mainloop stage raises shared memory enough to reduce residency from two CTAs/SM to one CTA/SM. It does not change the 232-register/thread pressure, doubles the occupancy loss, and increases scheduler starvation. This is a structural regression on SM86, not timing noise.

Valid profiler artifacts:

- `test_results/local3070_ncu2025_qkv_swizzle8_full_2026-07-22.ncu-rep`
- `test_results/local3070_ncu2025_qkv_stage4_clean_full_2026-07-22.ncu-rep`
- `test_results/local3070_ncu2025_ff1_swizzle8_full_2026-07-22.ncu-rep`
- `test_results/local3070_ncu2025_ff1_stage4_clean_skip2_full_2026-07-22.ncu-rep`

The failed/incorrect NCU-filter attempts are intentionally excluded from the comparison.

## Rollback verification

The stages=4 enum, parser surface, tests, and CUDA instantiations were removed. The dual SM75/SM86 build passed, the focused GEMM policy test passed, Docker CTest passed 18/18, and the post-rollback production smoke retained the canonical score SHA at 8.0548 ms.

Verification logs:

- build and focused test: `.gpu_queue/logs/e51efd2785e2.log`
- pre-gate idle: `.gpu_queue/logs/ff1ec87c547f.log`
- full CTest and exact smoke: `.gpu_queue/logs/afcb8595a079.log`
