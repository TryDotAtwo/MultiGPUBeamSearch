# RTX 3070 transformer GEMM follow-up gates — 2026-07-22

## Decision

Keep the accepted RTX 3070 production bundle unchanged:

- QKV and FF1: `m128n128`, stages 3, identity swizzle 8.
- Attention-output and FF2: `m128n128`, fused exact residual epilogue, identity swizzle 2.
- LayerNorm: row policy.
- FMHA: `q64k64 + padded64`.

The follow-up candidates remained byte-exact but failed the acceptance gate and were removed from production code: LayerNorm-to-FF1 mainloop fusion, QKV stages 2, residual GEMM `m128n256`, and the smaller `m64n64`/`m64n128` residual tile family.

## Measurement validity

A first `m128n256` timing/profile series overlapped with an interactive game on the RTX 3070 and is explicitly invalid for performance decisions. It showed 8–16.7 ms full-path variation and one anomalous 10.00 ms FF2 profile.

Before the accepted clean series:

- the queue container was restarted after its worker exited on a mounted-filesystem I/O/status-file error;
- a five-sample idle check reported no compute processes and `SM=0%`, `MEM=0%` throughout;
- all subsequent GPU work ran sequentially through the long-lived queue;
- NCU was explicitly `/opt/nvidia/nsight-compute/2025.1.1/ncu`.

Idle evidence: queue log `.gpu_queue/logs/099fdff9b308.log`.

## Exactness contract

The canonical complete score-dump SHA-256 remained:

`a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`

No `CUDA_LAUNCH_BLOCKING`, fallback, or production MLP change was introduced.

## LayerNorm-to-FF1 mainloop fusion

The initial integrated normalization differed from the canonical dump because the compiler contracted multiply-plus-add into FFMA. Using explicit `__fmul_rn` followed by `__fadd_rn` restored the original arithmetic sequence and byte-exact output.

The exact candidate was rejected:

| Metric | Materialized LayerNorm | Mainloop LayerNorm | Change |
|---|---:|---:|---:|
| FF1 duration | 1.04704 ms | 1.27328 ms | +21.6% |
| Registers/thread | 232 | 255 | worse |
| Executed instructions | 32,790,144 | 56,950,272 | +73.7% |
| Achieved occupancy | 16.40% | 16.36% | neutral |

The mainloop repeats LayerNorm work for every N tile. A wider 128x256 FF1 tile remained exact but worsened the full path to 8.9450 ms. All mainloop experimental code and policy surface were removed.

Artifacts:

- `test_results/local3070_ncu2025_ff1_materialize_full_2026-07-22.ncu-rep`
- `test_results/local3070_ncu2025_ff1_mainloop_full_2026-07-22.ncu-rep`
- queue exact logs `.gpu_queue/logs/5cbe5d172d34.log`, `.gpu_queue/logs/3636a746308c.log`, and `.gpu_queue/logs/cf9b3090c766.log`.

## QKV stages 2

QKV stages 2 preserved the canonical dump but reduced neither latency nor the limiting resource regime:

| Metric | Stages 3 | Stages 2 | Change |
|---|---:|---:|---:|
| QKV duration | 693.408 us | 705.664 us | +1.77% |
| Shared memory | 49,152 B | 32,768 B | lower |
| Registers/thread | 232 | 252 | worse |
| Achieved occupancy | 16.29% | 16.30% | neutral |
| Executed instructions | 10,883,808 | 11,045,376 | +1.48% |

The register increase cancelled the shared-memory reduction, so no additional CTA became resident. The stages-2 QKV instantiation and policy were removed.

Artifacts:

- `test_results/local3070_ncu2025_qkv_swizzle8_full_2026-07-22.ncu-rep`
- `test_results/local3070_ncu2025_qkv_stages2_full_2026-07-22.ncu-rep`
- queue exact log `.gpu_queue/logs/ea6afac58e62.log`.

## Residual GEMM m128n256

The candidate covered AttentionOut and FF2, whose output width is exactly 256. It halved the N-axis grid and reduced repeated A reads, while keeping each output's K-accumulation order exact.

Clean NCU 2025.1.1 results:

| Kernel | m128n128 | m128n256 | Change |
|---|---:|---:|---:|
| AttentionOut duration | 261.15 us | 344.29 us | +31.8% |
| FF2 duration | 959.74 us | about 1.10 ms | about +14.6% |
| AttentionOut DRAM throughput | 37.55% | 28.81% | lower |
| FF2 DRAM throughput | 25.56% | 18.95% | lower |
| AttentionOut executed instructions | 4,783,392 | 4,492,896 | lower |
| FF2 executed instructions | 10,932,768 | 9,976,416 | lower |
| Block size / grid size | 128 / 408 | 256 / 204 | changed |
| Registers/thread | 232 | 224 | lower |
| Theoretical occupancy | 16.67% | 16.67% | unchanged |

The wider CTA reduced instruction and input-read work, but did not increase occupancy and lowered effective IPC/memory utilization.

A clean, warm, alternating exact A/B20 confirmed the end-to-end regression:

| Variant | Median | Mean | Paired wins |
|---|---:|---:|---:|
| Production m128n128 | 8.10875 ms | 8.13688 ms | 20/20 |
| m128n256 both residual families | 8.58590 ms | 8.59443 ms | 0/20 |

The candidate regressed median latency by 5.88%. All 40 complete dumps matched the canonical SHA. The `m128n256` enum, parser, tests, and CUDA instantiation were removed.

Artifacts:

- `test_results/residual_m128n256_clean_exact_ab20/`
- `test_results/local3070_ncu2025_residual_baseline_clean_skip1_full_2026-07-22.ncu-rep`
- `test_results/local3070_ncu2025_residual_baseline_clean_skip3_full_2026-07-22.ncu-rep`
- `test_results/local3070_ncu2025_attn_out_n256_clean_full_2026-07-22.ncu-rep`
- `test_results/local3070_ncu2025_ff2_n256_clean_full_2026-07-22.ncu-rep`
- queue A/B log `.gpu_queue/logs/735dc302c996.log`.

## Residual GEMM small tiles

All measurements below were taken after a fresh idle check showed no compute processes and three consecutive `SM=0%`, `MEM=0%` samples. Every complete score dump matched the canonical SHA.

NCU showed why small CTAs were worth testing: `m64n64`, stages 3, swizzle 1 reduced the isolated AttentionOut kernel from 261.15 us to 243.74 us and FF2 from 959.74 us to 892.16 us, while registers fell from 232 to 102 and waves/SM doubled from about 5.1 to 10.2. However, swizzle 1 made the aggregate residual group 14.2% slower in Nsight Systems and regressed the full path.

Retuning the swizzle recovered most of that loss:

| Variant | Median | Paired wins | Change vs baseline |
|---|---:|---:|---:|
| Production baseline | 8.06120 ms | - | - |
| `m64n64`, stages 3, swizzle 4 for both families | 7.88215 ms | 17/20 | +2.221% |
| AttentionOut `m64n128` stages 2 + FF2 `m64n64` stages 3 | 7.87755 ms | 19/20 | +2.278% |
| AttentionOut `m64n128` stages 3 + FF2 `m64n64` stages 3 | 7.88430 ms | 17/20 | +2.194% |
| AttentionOut `m64n64` + FF2 `m64n128` stages 3 | 7.91040 ms | 20/20 | +1.871% |
| AttentionOut `m64n64` + FF2 `m64n128` stages 2 | 7.91680 ms | 19/20 | +1.791% |

A dedicated stage-count matrix also rejected `m64n64` stages 2: 8.07220 ms versus 7.97020 ms for stages 3 in that series. The best clean result, +2.278%, remained below the predeclared 3% median acceptance threshold. Keeping it would add CUTLASS instantiations and autotune/cache surface for a marginal hardware point, so the complete small-tile branch was removed.

Artifacts:

- NCU: `test_results/local3070_ncu2025_residual_m64_skip1_full_2026-07-22.ncu-rep` and `test_results/local3070_ncu2025_residual_m64_skip3_full_2026-07-22.ncu-rep`.
- Nsight Systems: `test_results/local3070_residual_baseline_nodes_clean_2026-07-22.*` and `test_results/local3070_residual_m64_nodes_clean_2026-07-22.*`.
- queue A/B logs `.gpu_queue/logs/9ffdbc6c0dd4.log`, `.gpu_queue/logs/d6a1543ce785.log`, `.gpu_queue/logs/0f616871da92.log`, `.gpu_queue/logs/e2f1d46473f9.log`, and `.gpu_queue/logs/60aadd9e06f4.log`.
## Production revalidation

After removing all rejected candidates:

- a fresh queue idle gate found no compute processes and three consecutive `SM=0%`, `MEM=0%` samples;
- Docker CTest passed 18/18;
- production smoke measured 8.0702 ms, consistent with the prior clean 8.0558 ms result;
- the complete score dump matched the canonical SHA;
- `git diff --check` passed.

Evidence: `.gpu_queue/logs/9c530c5c0e50.log`, `.gpu_queue/logs/73fc0e01cee2.log`, `.gpu_queue/logs/bb765c02f2c4.log`, and `test_results/production_after_small_tile_reject_2026-07-22.bin`.