# RTX 3070 transformer optimization saturation report (2026-07-22)

## Decision

The profiler-supported RTX 3070 optimization contour is saturated under the project's byte-exact and 3% acceptance gates. The selected production bundle remains the final local SM86 policy, and subsequent work moves to an independent SM75 / 2xT4 profile rather than transferring SM86 choices by assumption.

## Selected production bundle

- QKV and FF1: CUTLASS `m128n128`, stages 3, identity swizzle 8.
- Attention-output and FF2: CUTLASS `m128n128`, stages 3, fused exact residual epilogue, identity swizzle 2.
- LayerNorm: exact row policy.
- FMHA: CUTLASS `q64k64 + padded64`.
- Launch geometry: `B_MICRO=512`, concurrency 1, ring slot 1.

The accepted bundle improved isolated Stream1 median by 4.73% with 30/30 paired wins and Stream1->2->3 median by 7.07% with 20/20 paired wins. Every accepted complete score dump matched the canonical byte sequence.

## Exhausted measured axes

- GEMM: per-family tile shapes, warp shapes, stage counts, swizzles, vectorized epilogues, wider and smaller residual tiles, and LayerNorm mainloop fusion.
- Attention: q64/q32 query tiles, exact/padded head-dimension policies, alignment, final CLS-only variants, packed Q/K/V, and an L2 persistence window.
- LayerNorm: row scheduling, split shared slots, dtype specialization, and materialization/mainloop boundaries.
- Scheduling: transformer microbatch/concurrency and valid Stream1->2->3 ring-slot combinations.

The final Nsys profile attributes 72.5% of kernel time to GEMMs, 11.7% to bias-round-LayerNorm, and 11.0% to FMHA. All remaining measured candidates either regressed, failed byte-exactness, or produced less than the predeclared 3% target. The isolated `256 x 2` launch geometry was the last candidate above 3%, but its full pipeline gain was only 0.720% with 4/8 paired wins.

## Transition criterion

A new RTX 3070 direction would now require a speculative custom kernel rewrite without counter evidence of an end-to-end gain above 3%. Existing attention counters bound the entire FMHA share to about 11%, while the tested tail, alignment, layout, and cache changes yielded at most sub-percent full-path gains. The next evidence-bearing step is therefore fresh 2xT4 profiling and hardware-specific SM75 tuning.

## Primary evidence

- `test_results/local3070_residual_epilogue_swizzle_bundle_gate_2026-07-22.md`
- `test_results/local3070_transformer_current_profile_2026-07-22.md`
- `test_results/local3070_gemm_followup_rejections_2026-07-22.md`
- `test_results/local3070_memory_followup_gates_2026-07-22.md`
- `test_results/local3070_ff1_small_tile_combined_gate_2026-07-22.md`
- `test_results/local3070_transformer_launch_geometry_gate_2026-07-22.md`
