# H200 native FP16 next boundary

Source audit on 2026-09-15, parent 1a0115dc; no new GPU measurements.

The compact57 generic graph executes three full-token blocks followed by the
specialized final CLS block. Per full block:

1. LN1 (first input LN can already be prepared) -> attention_context.
2. QKV FP16 TMA/WGMMA -> qkv.
3. Fused attention -> attention_context (QKV dies after consumption).
4. Attention-output SM80 GEMM + residual -> tokens.
5. Bias + LN2 -> attention_context; persistent residual stored separately.
6. FF1 FP16 TMA/WGMMA + bias + ReLU -> ff_hidden.
7. FF2 SM80 GEMM + residual -> tokens; FF2 bias consumed by next LN.

Verified in stream1_transformer_forward generic loop and
stream1_transformer_linear_residual_cuda: residual GEMMs select Sm80 even on
H200. Current m128n128 FF2 uses tile128x128x32 and warp64x64x32.
There is no separate residual-add kernel at this boundary to simply remove.
The full FP16 hidden tensor remains necessary to the existing implementation.

Next bounded candidate: native SM90 FP16 FF2 residual GEMM, retaining global
hidden as the immediate control. Do not simultaneously change LN reduction,
rounding, hidden storage, attention, and scheduling. Preserve existing final
CLS handling and one device representation per operator. Prove weight layout
and in-place C/D legality with independent tail oracles before integration.
Do not fuse FF2 bias before its existing rounding boundary.

Gates: independent FP16/FP32 arithmetic oracle including tails; memcheck and
synccheck; full-model score comparison; seven alternating full Stream1 trials
against micro384 x8 FP16 TMA QKV/FF1, epilogue128x64, compact57, final-CLS split,
dual-input-LN off; then same 100M integrated pipeline profile if it wins.
Production default must not change on component timings alone.

The old h200_single_profile.sh omits the latest compact57/TMA/epilogue profile
and defaults to micro192 x12. It is not a valid fresh trace command for the
current fastest FP16 control without explicit correction.

## Isolated prototype gate

Subsequently resumed existing instance51120383 (H200,143771MiB), no new rental.
Source a6f07347 compiled with nvcc12.8, sm_90a, CUTLASS3.9.2. Independent exact
dyadic CPU oracle passes every output for rows1/31/128/131, K1024,N256,
nonzero residual. Physical memcheck and synccheck each report0errors; logs
fp16_residual_memcheck.log and fp16_residual_synccheck.log are retained.
These are component correctness gates only: no timing, model integration,
arbitrary-model numerical qualification or integrated speedup is established.
Restarted five-hour stop guard PID851 and verified it live after startup.

## Paired FF2 component timing

Source50a2911b, same H200: SM80 m128n128 residual GEMM versus SM90 prototype,
M21888 K1024 N256, seven alternating pairs,300 graph replays per measurement.
Median49.3981us versus29.6685us (1.665x). All pairs favor SM90.
Full outputs match exactly on the benchmark's deterministic dyadic input.
Each mode begins with the same residual; replays repeatedly update it in place.
Separate row/column-major weight buffers are benchmark controls only, not a
production two-copy layout. No transfer or offline packing included in timing.
Raw results: fp16_residual_benchmark.log. Model-level and pipeline gates remain
open; this result alone does not change defaults.
