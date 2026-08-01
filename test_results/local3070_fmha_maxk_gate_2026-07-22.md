# RTX 3070 CUTLASS FMHA max-K gate (2026-07-22)

## Candidate

The fixed transformer backend requires `head_dim=32`, while CUTLASS FMHA was instantiated with the conservative `kMaxK=64`. Added a fail-closed full-attention policy: `padded64` (default) and `exact32` (shape-specialized candidate).

## Correctness

- `padded64` versus `exact32`: all 40 complete dumps matched SHA-256 `a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`.
- `q64k64+padded64` versus `q32k64+exact32`: all 40 complete dumps matched the same SHA.
- Unknown max-K policy values throw; policy contracts execute even in Release builds.
- Full Docker CTest passed 18/18.

## Paired Stream1 timing

At `b_micro=512`, concurrency 1, CUDA Graph, selected `m128n128` GEMMs:

| Candidate | Baseline median | Candidate median | Delta | Pair wins |
|---|---:|---:|---:|---:|
| `q64k64+exact32` | 8.5067 ms | 8.4461 ms | +0.72% | 12/20 |
| `q32k64+exact32` | 8.5862 ms | 8.5966 ms | -0.12% | 10/20 |

All changes are below the 3% acceptance threshold.

## Nsight Compute 2025.1.1

For the q64 full FMHA kernel:

| Metric | padded64 | exact32 |
|---|---:|---:|
| Duration | 370.34 us | 362.43 us |
| Registers/thread | 125 | 124 |
| Dynamic shared/block | 18.94 KiB | 18.94 KiB |
| Achieved occupancy | 32.32% | 32.35% |
| No eligible | 40.77% | 41.11% |

The shape-specialized bound makes the isolated kernel about 2.14% faster, but saves only one register and no shared memory, so the full Stream1 benefit is too small.

## Decision

Keep `padded64` as production default. Retain `exact32` as an explicit fail-closed policy candidate for future composition, but do not cache/select it on RTX 3070. Keep the independently validated `q64k64` tile.

Artifacts:

- `test_results/fmha_maxk_ab_exact20/`
- `test_results/fmha_q32_maxk32_ab20/`
- `test_results/local3070_ncu2025_fmha_exact32_full_2026-07-22.ncu-rep`
