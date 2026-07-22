# RTX 3070 FF1 small-tile and combined residual gate (2026-07-22)

## Decision

Keep the accepted production transformer policies unchanged. The best combined candidate was byte-exact and won 19/20 pairs, but improved the full Stream1 median by only 2.744%, below the predeclared 3% acceptance gate. All temporary CUTLASS instantiations and policy surface were removed.

Production remains QKV/FF1 `m128n128`, stages 3, swizzle 8; AttentionOut/FF2 `m128n128`, stages 3, fused exact residual epilogue, swizzle 2; LayerNorm row; FMHA `q64k64+padded64`.

## Measurement validity

The first 60-dump swizzle matrix (`.gpu_queue/logs/5dd0db4ff372.log`) is excluded completely. Although its immediate pre-check was idle, an interactive graphics workload appeared during the run: invocations jumped to 40.87 and 51.39 ms and the job took 498.6 seconds. These timings are not used below.

Accepted evidence was split into short queue jobs. Every batch performed an immediate three-sample `nvidia-smi dmon` pre-check and a post-check; all decision batches were idle before launch and returned idle after launch. Clean swizzle logs: `.gpu_queue/logs/b9ce918958b8.log`, `.gpu_queue/logs/1c58250a5aae.log`, `.gpu_queue/logs/c26994d058ca.log`, `.gpu_queue/logs/5d8fe5dbeedb.log`. Combined logs: `.gpu_queue/logs/d9621c95d472.log`, `.gpu_queue/logs/25162efd95de.log`, `.gpu_queue/logs/ba700b61292d.log`, `.gpu_queue/logs/fddd0d829ae6.log`, `.gpu_queue/logs/52a8c8cfb14b.log`, `.gpu_queue/logs/b5309e7f9e88.log`, `.gpu_queue/logs/cf4975281d94.log`.

## Exactness

Canonical complete score-dump SHA-256: `a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`.

- FF1 swizzle matrix: 40/40 complete dumps exact.
- Combined FF1/residual gate: 80/80 complete dumps exact.
- Post-rollback production dump: exact.

No `CUDA_LAUNCH_BLOCKING`, fallback, or production MLP change was used.

## FF1 resource follow-ups

`m64n128`, warp 32x64, stages 2 reduces shared memory to 24.58 KiB and raises achieved occupancy from about 16.4% to 24.6%. NCU 2025.1.1 measured 968.83 us versus about 1.04 ms for production, but the full-path gain remained small.

| FF1 policy | Median | Change vs production | Paired wins |
|---|---:|---:|---:|
| production `m128n128/s3/sw8` | 8.12850 ms | - | - |
| `m64n128/s2/sw1` | 8.20525 ms | +0.944% | 2/8 |
| `m64n128/s2/sw2` | 8.07910 ms | -0.608% | 4/8 |
| `m64n128/s2/sw4` | 7.97205 ms | -1.925% | 6/8 |
| `m64n128/s2/sw8` | 8.02055 ms | -1.328% | 6/8 |

NCU artifact: `test_results/local3070_ncu2025_ff1_m64n128_stage2_full_2026-07-22.ncu-rep`.

The warp-32x64/stages-2/vector-4 epilogue experiment was exact but slower: 9.2471 ms versus 8.0594 ms; NCU measured 1.66 ms, 150 registers/thread, 32.77 KiB shared, and 16.67% occupancy. Artifact: `test_results/local3070_ncu2025_ff1_warp64x32_stage2_vector4_full_2026-07-22.ncu-rep`.

## Combined candidate

The candidate combined FF1 `m64n128/s2/sw4`, AttentionOut `m64n128/s2/sw4`, and FF2 `m64n64/s3/sw4`, keeping the fused exact residual epilogue.

| Variant | Median | Mean | Change vs baseline median | Paired median | Wins |
|---|---:|---:|---:|---:|---:|
| production | 8.06680 ms | 8.07445 ms | - | - | - |
| FF1 small tile only | 7.92095 ms | 7.93014 ms | -1.808% | -1.819% | 20/20 |
| residual small tiles only | 7.88465 ms | 7.89302 ms | -2.258% | -2.300% | 20/20 |
| combined | 7.84545 ms | 7.86266 ms | -2.744% | -2.824% | 19/20 |

The effects overlap rather than add, so the combined path remains below 3% and is rejected.

## Rollback verification

After removing the experimental policies/tests/instantiations, Docker CTest passed 18/18, production measured 8.0521 ms, the complete dump matched the canonical SHA, and `git diff --check` passed. Evidence: `.gpu_queue/logs/1b4abf145395.log` and `test_results/production_after_ff1_residual_reject_2026-07-22.bin`.