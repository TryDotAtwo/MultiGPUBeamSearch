# RTX 3070 transformer memory follow-up gates — 2026-07-22

## Decision

Keep the accepted schema-v3 RTX 3070 production policy unchanged. Every runnable candidate below preserved the canonical complete score-dump SHA-256 `a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`, but none passed the speed/consistency gate.

## QKV/FF1 swizzle 4

All 80/80 dumps matched exactly. Twenty rotated repeats per configuration gave:

| QKV swizzle | FF1 swizzle | Median, ms | Mean, ms |
|---:|---:|---:|---:|
| 8 | 8 | 8.06155 | 8.08748 |
| 4 | 8 | 8.06615 | 8.11181 |
| 8 | 4 | 8.06385 | 8.09719 |
| 4 | 4 | 8.06250 | 8.08499 |

No swizzle-4 combination improved the selected 8/8 median. Evidence: `.gpu_queue/logs/a39aa5c60664.log`.

## LayerNorm dtype specialization

Explicit FP16 specialization with round-to-nearest preserved 60/60 dumps but was within noise: runtime-dispatch median 8.05980 ms versus specialized 8.05785 ms (0.024%). The specialization was removed. Evidence: `.gpu_queue/logs/a85c24347a6c.log`, `.gpu_queue/logs/b32f3162d42d.log`.

## FMHA memory axes

NCU SourceCounters for selected q64k64 showed that the largest shared-memory excessive-wavefront sites are the Q/K/V `cp.async` loads and the softmax shared atomics. Baseline selected FMHA: 370.34 us, 26,999,420 instructions, 125 registers/thread, 32.32% achieved occupancy, 515,005 excessive shared wavefronts, and 188,347 bank conflicts.

- `q64k32` was not compiled: CUTLASS correctly rejected the configuration because `kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize` is required and K32 makes the boundary equal. The vendor invariant was not bypassed.
- `q64k64v4` (four-element alignment) preserved 40/40 dumps but regressed padded64 median from 8.05605 to 8.16980 ms. NCU measured 419.14 us, 27,884,156 instructions, 1,815,213 excessive shared wavefronts, and 202,830 bank conflicts. It remains compiled only as an opt-in future-hardware axis.
- `exact32` preserved 60/60 dumps in the final A/B30. Median was 8.30255 ms versus 8.32665 ms for padded64 (+0.289%), with only 17/30 paired wins. It remains an opt-in candidate and is not cached.

Evidence: `test_results/local3070_ncu2025_selected_fmha_full_2026-07-19.ncu-rep`, `test_results/local3070_ncu2025_fmha_v4_full_2026-07-22.ncu-rep`, `.gpu_queue/logs/ed739a5ac056.log`, `.gpu_queue/logs/a1063d2e7643.log`, `.gpu_queue/logs/28e573059f29.log`.

## L2 persistence window

A benchmark-only CUDA access-policy window pinned the 13,369,344-byte attention-context buffer with the RTX 3070 maximum 2,883,584-byte persisting-L2 set-aside and hit ratio 0.215686. All 40/40 dumps matched exactly, but the policy regressed median from 8.05555 to 8.14735 ms (about -1.14%) and won only 1/20 pairs. The experiment was removed completely; post-rollback build and exact smoke retained the canonical SHA at 8.0660 ms.

Evidence: `.gpu_queue/logs/2fe26025e940.log`, `.gpu_queue/logs/326d7ef54a50.log`.

## Next optimization boundary

Small FMHA tiling/alignment/cache-window axes are exhausted on this RTX 3070. Nsys still attributes 72.5% of kernel time to GEMMs and about 11.7% to bias-round-LayerNorm. The next material path is an opt-in exact LayerNorm-to-QKV/FF1 CUTLASS mainloop fusion: compute row statistics with the existing reduction order, consume the unmaterialized row plus statistics in the GEMM mainloop, and accept only after the complete score dump is byte-identical and paired speed gates pass.