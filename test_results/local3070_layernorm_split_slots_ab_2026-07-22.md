# RTX 3070 LayerNorm split-slot A/B gate (2026-07-22)

## Hypothesis

The specialized 256-wide LayerNorm kernels reused shared slot 0 for warp partials, mean, and inverse standard deviation. Reserving independent slots for mean and inverse standard deviation removes the post-mean block barrier without changing reduction order or output arithmetic.

## Correctness

- Baseline and split-slot builds produced the same complete score dump in 20 alternating measured pairs plus two warmups: 42/42 files had SHA-256 `a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`.
- The split-slot candidate passed bounded Compute Sanitizer racecheck at `b_micro=8`: `0 hazards displayed (0 errors, 0 warnings)`.
- After restoring baseline as the production default, a fresh production dump retained the same SHA-256.
- Full Docker CTest passed 18/18.

## RTX 3070 paired timing

Configuration: native FP16 Stream1 transformer, `b_micro=512`, concurrency 1, CUDA Graph, final CLS-only output, selected `m128n128` GEMM policies. Order alternated on every pair.

- Baseline median: `8.3263 ms`.
- Split-slot median: `8.3432 ms`.
- Median speedup: `-0.20%`.
- Paired mean baseline-minus-split: `-0.002445 ms`.
- Split-slot pair wins: `9/20`.

The earlier unpaired split-slot median (`8.64865 ms` versus an older `8.9312 ms` reference) was GPU-state drift, not a reproducible optimization.

## Nsight Compute 2025.1.1

One identical `stream1_transformer_layernorm256_copy_kernel` launch was profiled from each build:

| Metric | Baseline | Split slots |
|---|---:|---:|
| Duration | 226.24 us | 224.19 us |
| Stall Barrier | 8.20 inst/warp | 7.84 inst/warp |
| No Eligible | 56.10% | 56.06% |
| Eligible warps/scheduler | 0.81 | 0.82 |
| Registers/thread | 20 | 20 |
| Achieved occupancy | 93.13% | 93.09% |

Removing one barrier improves this isolated kernel by about 0.91%, but does not produce a measurable family/end-to-end gain. Remaining latency is dominated by the other reductions/barriers, and the change is far below the 3% acceptance threshold.

## Decision

Rejected for production. The safe slot-0 implementation with the required post-mean barrier remains the default. The split-slot implementation remains compile-time opt-in and contract-tested only, so it can be revisited in a larger fused LayerNorm/residual design. No full-pipeline benchmark was run because the candidate failed the isolated paired gate.

Artifacts:

- `test_results/ln_ab_exact20/`
- `test_results/local3070_layernorm_split_slots_racecheck_2026-07-22.log`
- `test_results/local3070_ncu2025_layernorm_baseline_full_2026-07-22.ncu-rep`
- `test_results/local3070_ncu2025_layernorm_split_slots_full_2026-07-22.ncu-rep`
