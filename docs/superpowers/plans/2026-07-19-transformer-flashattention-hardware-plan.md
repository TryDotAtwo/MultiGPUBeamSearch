# Fixed-shape transformer attention hardware plan

## Goal

Reduce Q/K/V global-memory traffic for the fixed Megaminx transformer while preserving every complete score byte and selecting policies by exact GPU/model signature.

## Measured starting point

- Full attention is CUTLASS fused FMHA, Q=K=51, eight heads, head_dim=32, q64k64, four warps/block.
- It accounts for about 10.4% of post-fix kernel time.
- q32k64 is exact but 2.56% slower because two query blocks reload K/V.
- The existing general CUTLASS Q=1 final-CLS mode was exact but slower on RTX 3070.
- NCU counters are currently blocked by the local driver/tool mismatch; use source resource bounds, Nsight Systems timings, CUDA function attributes, exact dumps, and paired sweeps until the driver/tool pair is updated.

## Implementation sequence

1. Add a dedicated SM80+ FP16 Q=1, K=51, D=32 CLS kernel: one block per batch/head, Q held in registers, coalesced K/V loads, warp-level max/sum, probabilities never written globally, direct context output. Compile it as an explicit policy; keep the existing path only as benchmark reference.
2. Sweep 1/2/4-warp mappings and vectorized half2/128-bit K/V loads. Reject any candidate that changes a complete score dump or fails to improve isolated final-layer attention by 3%.
3. Benchmark the accepted CLS candidate in full Stream1 and alternating Stream1-to-2-to-3 pairs. Do not select it if pipeline median regresses.
4. For full 51x51 attention, build a single-K/V-load custom tile with 51 valid Q rows, Q fragments and online-softmax state in registers, K/V in shared memory, and no global score/probability materialization. Candidate warp/stage counts are compiled policies, not guessed runtime heuristics.
5. Only after attention is independently faster, evaluate a QKV GEMM epilogue that writes the head-major layout consumed coalescently by the custom kernel. Accept on combined QKV+attention time, not attention alone.
6. Extend the exact-signature cache with independent full-attention and CLS-attention fields. Any malformed/stale entry resolves every attention field to the measured reference policy.

## Gates

For each candidate: warmup; at least 20 independent processes; byte comparison of every complete dump to the baseline oracle; strict unknown-policy rejection; focused tests; full CTest; bounded racecheck; isolated Stream1; alternating full pipeline. No CUDA_LAUNCH_BLOCKING and no runtime fallback.