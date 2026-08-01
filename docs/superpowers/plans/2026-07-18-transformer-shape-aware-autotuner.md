# Shape-Aware Transformer GEMM Autotuner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline/first-run FF1 autotuner that selects a mathematically exact CUTLASS policy for each transformer shape and GPU, caches the decision, and otherwise falls back to the established baseline.

**Architecture:** The benchmark exports complete `uint32_t score_key` buffers for exact byte comparison. A small Python controller builds a signature from GPU identity, compute capability, dtype, GEMM `M/N/K`, concurrency, and fused epilogue; it runs each eligible policy repeatedly, rejects any candidate whose complete output ever differs from baseline, selects by median latency with a minimum margin, and atomically writes a versioned JSON cache. Production selection remains fail-closed: an absent, stale, or mismatched cache cannot override baseline.

**Tech Stack:** C++17, CUDA 12.8, CUTLASS, Python 3, JSON, Docker, CMake/CTest, RTX 3070 SM86.

## Global Constraints

- Use `gpu-dev-cutlass-nsight:2026-05-24` and real FP16 p900 weights locally.
- Never accept checksum/digest alone as the correctness gate; compare every output byte on every measured repetition.
- Cache keys include GPU name, compute capability, dtype, `M/N/K`, concurrency, epilogue, model fingerprint, and tuner schema version.
- Unset/missing/malformed/mismatched cache means baseline; unknown policy names fail closed.
- A candidate needs at least five exact repetitions and at least 3% median improvement; default validation uses ten repetitions.
- Keep SM75 and BF16 on baseline until independently tuned. Do not modify Stream 4.

---

### Task 1: Exact score dump contract

**Files:**
- Modify: `tools/stream_benchmark_transformer.cu`
- Test: `tests/stream1_transformer_gemm_policy_tests.cpp`

**Interfaces:**
- Consumes: optional `BEAM_STREAM1_TRANSFORMER_SCORE_DUMP` path.
- Produces: one binary file containing a fixed header followed by all lane-ordered `uint32_t score_key` values.

- [ ] Add a failing unit test for dump metadata validation helpers: zero lanes, zero values, and arithmetic overflow must be rejected.
- [ ] Implement a small header-only dump contract with schema magic, version, lane count, values per lane, and checked total size.
- [ ] Add optional binary emission after device-to-host copies; opening or writing the requested path must fail loudly.
- [ ] Build the focused test and benchmark, then produce two baseline dumps and require byte equality.

### Task 2: Signature, measurements, and cache controller

**Files:**
- Create: `tools/stream1_transformer_autotune.py`
- Create: `tests/test_stream1_transformer_autotune.py`

**Interfaces:**
- Consumes: benchmark executable, candidate policy list, model/benchmark environment, repeat count, and cache path.
- Produces: versioned JSON `{signature, selected_policy, baseline_median_ms, selected_median_ms, evidence}`.

- [ ] Write failing Python tests for canonical signature serialization, exact dump rejection, median selection, the 3% threshold, nondeterministic candidate rejection, and malformed-cache fallback.
- [ ] Implement pure selection and cache functions without launching GPU work.
- [ ] Implement subprocess execution with one warm-up plus ten measured runs per policy, unique dump paths, captured stdout, and GPU metadata from `nvidia-smi`.
- [ ] Compare each candidate dump byte-for-byte with the baseline dump from the same fixed workload; reject on the first mismatch and retain evidence.
- [ ] Write cache through a temporary sibling file followed by atomic replace only after all gates pass.

### Task 3: Fail-closed cache consumption

**Files:**
- Modify: `cuda/stream1_transformer_gemm_policy.hpp`
- Modify: `tests/stream1_transformer_gemm_policy_tests.cpp`
- Create: `tools/stream1_transformer_policy_from_cache.py`

**Interfaces:**
- Consumes: cache plus current signature.
- Produces: validated policy name for the launcher; failure returns `baseline` with a reason.

- [ ] Add failing tests covering schema mismatch, GPU mismatch, shape mismatch, model mismatch, unknown policy, and valid exact match.
- [ ] Implement strict cache parsing and signature equality in the launcher helper.
- [ ] Keep the CUDA parser limited to compiled policy names and baseline-by-default; never silently map an unknown cached value.
- [ ] Verify every invalid case selects baseline before process launch and every direct unknown CUDA policy still throws.

### Task 4: Local RTX 3070 verification

**Files:**
- Create: `test_results/local3070_transformer_shape_autotune_2026-07-18.md`
- Modify: `memory/CHANGELOG.md`
- Modify: `memory/PROMPTS.md`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: correctness/speed evidence and a reusable RTX 3070 cache or a documented baseline decision.

- [ ] Run focused CPU/Python tests and the complete Docker CTest suite.
- [ ] Run the autotuner for puzzle 0, `b_micro=512`, concurrency 2, FP16 p900, block51 final-CLS mode.
- [ ] Confirm the known rare `m64n128` mismatch is rejected whenever observed; if ten runs do not reproduce it, increase to fifty before allowing selection.
- [ ] Run five final default and Stream1-3 pipeline repetitions using only the selected verified policy.
- [ ] Record GPU/driver/CUDA/build/signature, every rejection reason, exact-output evidence, medians, and next candidate family.
