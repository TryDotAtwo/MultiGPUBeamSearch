# RTX 3070 residual epilogue and swizzle bundle gate — 2026-07-22

## Decision

Accept the exact-signature RTX 3070 bundle in `local3070_transformer_policy_v3.json`:

- QKV and FF1: `m128n128`, stages 3, identity swizzle 8.
- Attention-output and FF2: `m128n128`, exact fused residual + fp16-round + bias epilogue, identity swizzle 2.
- LayerNorm: exact row policy.
- FMHA: the previously selected `q64k64 + padded64` policy.

This cache is bound to the exact RTX 3070 Laptop GPU/driver/model/workload signature. It is not an A100 or T4 default.

## Mathematical correctness

The canonical complete score-dump SHA-256 remained:

`a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`

- Attention-output/FF2 four-way swizzle sweep: 80/80 complete dumps matched.
- Prior-production versus selected bundle: 60/60 complete dumps matched.
- Selected bundle bounded racecheck at `b_micro=8`: `0 hazards displayed (0 errors, 0 warnings)`.
- No `CUDA_LAUNCH_BLOCKING`, fallback, or MLP production change was introduced.

## Residual swizzle sweep

Twenty rotated repeats per configuration were measured after the exact fused epilogue was enabled.

| Configuration | Median, ms | Mean, ms | Wins vs swizzle 1/1 |
|---|---:|---:|---:|
| attention 1, FF2 1 | 8.36345 | 8.32887 | reference |
| attention 2, FF2 1 | 8.26075 | 8.28684 | 13/20 |
| attention 1, FF2 2 | 8.13430 | 8.24530 | 15/20 |
| attention 2, FF2 2 | 8.18790 | 8.23161 | 19/20 |

The combined swizzle 2/2 policy had the strongest paired consistency. Its selection is additionally supported by the profiler and by the final bundle gate.

## Nsight evidence

Nsight Systems node traces measured the fused residual GEMM group:

| Metric | Swizzle 1 | Swizzle 2 |
|---|---:|---:|
| Group total | 16.816672 ms | 16.080468 ms |
| Instances | 56 | 56 |
| Improvement | — | 4.38% |

Nsight Compute 2025.1.1 showed the intended memory-reuse mechanism without a resource trade-off:

| Kernel | DRAM swizzle 1 | DRAM swizzle 2 | Registers/thread | Achieved occupancy |
|---|---:|---:|---:|---:|
| Attention-output | 52.56% | 38.98% | 232 | 15.88–15.89% |
| FF2 | 38.02% | 25.90% | 232 | 15.86–15.87% |

Thus swizzle 2 reduces DRAM pressure while preserving the compiled kernel's register and occupancy footprint.

## Final bundle A/B

Thirty alternating exact pairs compared the prior production policy to the complete selected bundle.

| Variant | Median, ms | Mean, ms | 10% trimmed mean, ms |
|---|---:|---:|---:|
| Prior production | 8.46125 | 8.525257 | 8.509288 |
| Selected bundle | 8.06095 | 8.088110 | 8.068900 |

Median improvement: **4.73%**. The selected bundle won **30/30** pairs and saved 0.437147 ms per process on average.

## Stream1 → Stream2 → Stream3 gate

After one discarded warmup pair, twenty alternating `stream123` pairs used b512, transformer micro 512, concurrency 1, one Stream3 slot, and eight graph jobs.

| Variant | Median, ms | Mean, ms |
|---|---:|---:|
| Prior production | 76.2467 | 76.7793 |
| Selected bundle | 70.8561 | 71.3048 |

The selected bundle improved the pipeline median by **7.07%**, won **20/20** pairs, and every run reported `status=OK`.

## Cache and verification

- Schema-v3 cache atomically validates all GEMM policies, swizzles, stages, residual epilogues, LayerNorm, and FMHA fields.
- Any missing, unknown, cross-field-incompatible, or signature-mismatched cache returns the complete conservative environment; no partial cache application is allowed.
- Python cache/autotune tests: 15/15 passed.
- Docker CTest: 18/18 passed.
- `git diff --check`: passed.

Primary artifacts:

- `test_results/local3070_transformer_policy_v3.json`
- `test_results/local3070_transformer_signature_v3.json`
- `test_results/local3070_bundle_exact_ab30/`
- `test_results/local3070_selected_bundle_racecheck_b8_2026-07-22.log`
- `.gpu_queue/logs/02fbf1efe0b0.log` (four-way exact sweep)
- `.gpu_queue/logs/1c45d194d937.log` (30-pair bundle gate)
- `.gpu_queue/logs/8ed29c8b2bb8.log` (20-pair pipeline gate)
- `test_results/local3070_residual_swizzle{1,2}_nodes_2026-07-22.nsys-rep`
- `test_results/local3070_ncu2025_residual_swizzle{1,2}_skip{1,3}_full_2026-07-22.ncu-rep`
