# Local RTX 3070 Transformer Inference Optimization Design

Date: 2026-07-18

## Goal

Improve the exact native CUDA/CUTLASS `piece_transformer` inference path on the
local RTX 3070 Laptop GPU in Docker. The local SM86 result is the development
target; A100 validation and final parameter selection are deferred until a
stable local winner exists. The MLP path and Stream 4 are out of scope.

## Correctness Contract

- Use the real exported FP16 p900 weights.
- Keep `block51=1`, `final_cls_only=1`, and `final_cls_attention=0` as the
  initial reference configuration.
- Every accepted change must preserve checksum `841858064`, digest
  `821400116975659197`, and the first score keys for the fixed input.
- Existing contract, dispatcher, and transformer CUDA tests must pass.
- No silent fallback to a generic, LibTorch, or PyTorch backend.

## Measurement Contract

- Reuse `gpu-dev-cutlass-nsight:2026-05-24` on the RTX 3070 Laptop SM86.
- Primary point: `b_micro=512`, `concurrency=2`, CUDA Graph benchmark.
- Compare medians from at least five warmed repetitions.
- Record temperature, clocks, power, memory, throughput, checksum, and digest.
- Validate the isolated winner with Stream1-3. Reject a candidate that regresses
  pipeline median by more than 2%.
- Treat improvements below 3% as noise and prefer at least 5%.

## Current Baseline And Profile

Three clean runs measured 1,079,201, 1,022,684, and 1,000,299 candidates/s.
The bounded Nsight Systems run measured 1,044,277 candidates/s with the same
checksum and digest.

GPU time:

- residual GEMMs: 26.8%;
- FF1 bias+SiLU GEMMs: 24.8%;
- QKV bias GEMMs: 20.4%;
- CUTLASS FMHA: 12.1%;
- bias+round+LayerNorm: 11.3%;
- input and remaining work: 4.6%.

Launch overhead is not the primary limiter. GEMM policy and epilogue memory
behavior are the first target.

## Approaches

### A. Per-operation GEMM policy tuning (recommended first)

Instantiate a small explicit set of SM80 FP16 policies separately for QKV,
FF1, attention-output residual, and FF2 residual. This targets 72% of measured
GPU time with low architectural risk. The candidate set must stay small to
control compile time and binary size.

### B. Split final-layer `Q_cls + KV_all`

Compute final Q only for CLS, retain K/V for all 51 tokens, and use a fixed
`Q=1, K=51, heads=8, head_dim=32` attention kernel. This removes exact
unnecessary work but changes only one quarter of the QKV family, so it follows
GEMM tuning. The generic CLS FMHA regression means a specialized kernel is
required.

### C. Fused transformer block

Fuse residual, bias, normalization, and projection boundaries more deeply.
This has the largest long-term traffic reduction and the highest correctness
risk. Previous local fusions were neutral or regressive, so it follows fresh
NCU evidence.

## Implementation Sequence

1. Add a failing deterministic contract for named per-operation GEMM policies.
2. Add a minimal SM80 candidate set, beginning with FF1.
3. Run correctness tests and five-repeat local sweeps.
4. Keep stable winners and remove losing policies.
5. Re-profile, then use NCU on the new dominant kernel.
6. Validate Stream1-3 before designing `Q_cls + KV_all`.

## Deliverables

- source and deterministic tests for accepted policy selection;
- raw reports and profiler artifacts under `test_results/`;
- a concise baseline/candidate/pipeline verification report;
- corresponding `memory/CHANGELOG.md` updates.
