# Native Transformer Memory-Traffic Optimization Design

Date: 2026-07-19
Target branch: `codex/stream1-piece-transformer`
Local target: RTX 3070 Laptop GPU (SM86)
Deployment target: NVIDIA A100 (SM80)

## Objective

Accelerate native CUDA/CUTLASS Stream1 transformer inference by reducing data movement and improving reuse in registers, shared memory, L1/L2, and tensor-core operand pipelines. The primary constraint is memory-system efficiency rather than scalar ALU throughput.

Every accepted optimization must preserve the complete score-key output byte-for-byte for a fixed deterministic workload. No fallback path, `CUDA_LAUNCH_BLOCKING`, approximation, distillation, or production MLP change is allowed.

## Current State

The p900 transformer has fixed hot shapes: sequence length 51, model width 256, eight heads, head dimension 32, FF width 1024, and four layers. The native path uses CUTLASS tensor-core GEMMs and a CUTLASS example-41 fused attention kernel instantiated as a 64x64x64 attention tile.

Existing profiles attribute roughly 77 percent of GPU time to CUTLASS GEMMs, 13 percent to LayerNorm/copy kernels, and 8 percent to attention. Sampled GEMMs use tensor cores but show substantially higher DRAM pressure than SM utilization. The current QKV tensor is interleaved with a 768-element token stride. The attention kernel avoids materializing the full attention matrix, but its tiling and data movement have not been specialized for the fixed 51x32 attention shape or separately tuned for SM86 and SM80.

A shared-memory LayerNorm race has been fixed. Baseline eager and CUDA Graph execution are now deterministic, so the exact-output autotuner can safely evaluate new policies.

## Optimization Strategy

Work proceeds in four gated phases. A later phase starts only after the preceding phase has measured evidence and a correctness-clean candidate or a documented negative result.

### Phase 1: Fresh decomposition and counters

Create a post-fix profile for these kernel families separately:

- QKV projection;
- full 51-query FMHA;
- final-layer CLS-only one-query FMHA;
- attention output projection;
- FF1 fused bias and SiLU;
- FF2 residual projection;
- LayerNorm and fused bias-LayerNorm.

Nsight Systems establishes per-family time and launch counts. Nsight Compute collects tensor utilization, achieved occupancy, registers per thread, shared-memory usage, L1/L2 hit rates, global-memory sectors, shared-bank conflicts, and dominant warp-stall reasons. Measurements use a warmed fixed workload and record GPU, driver, clocks when available, CUDA version, build flags, batch, concurrency, and memory consumption.

The first candidate gate reruns the existing baseline and `m64n128` FF1 policies after the race fix. The old candidate failures are not treated as policy failures because the shared LayerNorm race affected both policies.

### Phase 2: Shape-specific GEMM policies

Replace the single shared transformer GEMM policy with independently selectable compiled policies for:

- QKV: M x 768 x 256;
- attention output: M x 256 x 256;
- FF1: M x 1024 x 256;
- FF2: M x 256 x 1024;
- final CLS projections where M is only the parent microbatch.

Candidate dimensions include threadblock shape, warp shape, pipeline stages, and alignment-compatible epilogue choice. The tuner changes one policy family at a time and keys its cache by GPU identity, compute capability, dtype, exact GEMM shape, batch, concurrency, epilogue, model fingerprint, and schema version.

A policy is accepted only when every full output dump matches the deterministic baseline and its median improvement is at least 3 percent without fragile memory or occupancy behavior.

### Phase 3: Fixed-shape FlashAttention-2-style kernels

Implement two attention families rather than forcing one general kernel to cover both workloads.

#### Full attention: Q=K=51, head dimension 32

The SM80+ kernel will:

- partition Q rows across warps to avoid unnecessary split-K work and cross-warp reduction;
- keep Q fragments and online-softmax state in registers;
- load coalesced K/V tiles into shared memory and reuse each tile across multiple Q rows;
- overlap global-to-shared copies with tensor-core work using `cp.async` double buffering where occupancy permits;
- compute QK, running max/sum normalization, and PV without writing scores or probabilities to global memory;
- use padding or swizzled shared layouts only when counters show bank conflicts;
- expose compiled Br, Bc, warp-count, and stage-count policies for SM86 and SM80 tuning.

Because sequence length is 51, candidates must account for tail overhead. Tiles are selected from measured behavior rather than assuming power-of-two 64x64 padding is optimal.

#### Final CLS attention: Q=1, K=51, head dimension 32

Use a separate small-query kernel. It loads or streams the 51 K/V rows once per head, holds the single Q and online-softmax state in registers, and directly emits one context vector. It does not launch or allocate work for the other 50 query rows.

The existing CUTLASS attention remains a benchmark reference during development, not a runtime fallback for an accepted policy.

### Phase 4: Layout and inter-kernel traffic

Measure and then optimize these boundaries:

1. QKV projection output layout to attention input;
2. attention context to output projection;
3. projection plus residual/bias to LayerNorm;
4. FF2 plus residual/bias to the following LayerNorm.

Evaluate a packed head-major Q/K/V layout against the current interleaved token-major layout. The preferred design makes the QKV GEMM epilogue write the exact layout consumed coalescently by attention, avoiding a separate transpose or pack kernel. A layout is accepted only if the QKV projection plus attention total improves; improving attention alone while making projection slower is insufficient.

Fusion candidates first remove global round trips at existing mathematical boundaries. Residual, bias, and LayerNorm ordering and FP16 rounding points must match the established block51 path exactly. Register pressure and occupancy are measured before retaining any wider fusion.

L2 reuse is treated as a scheduling and working-set property, not as explicit software-managed storage. Sweeps test batch and concurrency because excessive concurrent scratch traffic can evict weights and reduce L2 locality. The selected point must accelerate both isolated Stream1 and the Stream1-to-2-to-3 pipeline.

## Autotuning and Policy Selection

The offline tuner owns candidate measurement and cache generation. Production consumes only a validated exact-signature cache and otherwise selects the compiled baseline policy. An absent, malformed, stale, mismatched, or unknown entry fails closed to baseline.

Attention and GEMM policies are independent fields. This prevents a good FF1 tile on one GPU from implicitly selecting an unsuitable attention tile. RTX 3070 and A100 produce separate caches even though both use SM80-class tensor-core instructions.

## Correctness and Safety Gates

For every candidate:

1. Run a warm-up outside timed samples.
2. Run at least 20 independent measured processes for the fixed workload.
3. Compare every byte of every complete score-key dump with the deterministic baseline.
4. Reject on any mismatch, CUDA error, timeout, racecheck error, or unstable hash.
5. Run Compute Sanitizer racecheck on the selected kernel implementation.
6. Run focused CUDA tests, full CTest, and autotuner unit tests.
7. Benchmark isolated Stream1 and the real Stream1-to-2-to-3 pipeline.

Checksums and digests remain useful diagnostics but are not the acceptance oracle. The complete binary dump is the oracle.

## Performance Acceptance

A local candidate must provide at least 3 percent median improvement in its targeted stage and must not regress the Stream1-to-2-to-3 median. Larger architecture changes such as a custom attention kernel should target a meaningful end-to-end gain rather than a sub-percent microbenchmark result.

Report median, dispersion, memory footprint, clocks when available, and all failed candidate rows. Select the fastest stable configuration with operating margin, not a point that barely fits memory or occupancy constraints.

## Scope Boundaries

- Optimize one-GPU Stream1 first; multi-GPU scheduling is unchanged.
- Do not modify production MLP behavior.
- Do not change Stream2, Stream3, or Stream4 mathematics or contracts.
- Do not introduce autoregressive KV caching; the cache work concerns hardware data reuse within one fixed-sequence inference.
- Keep SM75/T4 and BF16 on independently verified policies; SM86 results do not authorize them.
- Validate on RTX 3070 first, then repeat profiling and tuning on A100 before selecting an A100 policy.

## Deliverables

- Post-fix Nsight Systems and Nsight Compute evidence.
- Independent GEMM policy families and exact-output autotuning results.
- Full-attention and CLS-only fixed-shape kernel experiments with negative results retained.
- Layout/fusion measurements and selected policies.
- Versioned RTX 3070 and later A100 cache artifacts.
- A final report comparing baseline and selected configurations for isolated Stream1 and Stream1-to-2-to-3.
