# Transformer Memory Profiling and Shape-Aware GEMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a deterministic post-fix memory-performance baseline and select independent exact CUTLASS policies for QKV, attention-output, FF1, FF2, and final-CLS GEMMs on RTX 3070 before writing a custom FlashAttention-2-style kernel.

**Architecture:** Extend the existing compiled-policy and exact-dump autotuner contracts from one FF1 selector to independent GEMM-family selectors. Profile the unchanged deterministic baseline first, sweep one family at a time, and accept only policies that preserve every score byte and improve both their targeted stage and the real Stream1-to-2-to-3 pipeline.

**Tech Stack:** C++17, CUDA 12.8, CUTLASS, Python 3, JSON, CMake/CTest, Nsight Systems, Nsight Compute, Compute Sanitizer, Docker, RTX 3070 SM86.

## Global Constraints

- Use `gpu-dev-cutlass-nsight:2026-05-24` with real FP16 p900 weights under `weights/megaminx_vlad_transformer_fp16`.
- Preserve complete score-key output byte-for-byte for every accepted policy; checksum and digest alone are not acceptance gates.
- Run normal asynchronous execution; do not use `CUDA_LAUNCH_BLOCKING` as a fix and do not add runtime fallbacks.
- Do not modify production MLP behavior or Stream2/3/4 mathematics.
- Warm every measured configuration and run at least 20 independent measured processes before acceptance.
- Require at least 3 percent median improvement in the targeted stage and no Stream1-to-2-to-3 median regression.
- Key policy caches by GPU identity, compute capability, dtype, exact shapes, batch, concurrency, epilogues, model fingerprint, and schema version.
- Tune RTX 3070 first; do not infer an A100 policy from SM86 measurements.

---

### Task 1: Deterministic post-fix baseline and kernel-family profile

**Files:**
- Create: `tools/stream1_transformer_profile_summary.py`
- Create: `tests/test_stream1_transformer_profile_summary.py`
- Create: `test_results/local3070_transformer_memory_profile_2026-07-19.md`
- Modify: `memory/CHANGELOG.md`

**Interfaces:**
- Consumes: Nsight Systems SQLite export with `CUPTI_ACTIVITY_KIND_KERNEL` rows and optional Nsight Compute CSV rows.
- Produces: `summarize_kernels(rows: Iterable[KernelRow]) -> list[KernelFamilySummary]` and a Markdown table with family, launches, total milliseconds, share, average microseconds, and matched kernel names.

- [ ] **Step 1: Write failing classifier and aggregation tests**

Create tests with synthetic names representing CUTLASS fused GEMM, plain GEMM, FMHA, LayerNorm, input-build, and score-quantize kernels. Assert that aggregation preserves total nanoseconds, sorts descending by total time, and places unknown names in `other` rather than dropping them.

```python
class ProfileSummaryTests(unittest.TestCase):
    def test_aggregation_preserves_time_and_classifies_unknown(self):
        rows = [
            KernelRow("cutlass::Kernel2", 3_000),
            KernelRow("attention_kernel_batched_impl", 2_000),
            KernelRow("stream1_transformer_layernorm256_copy_kernel", 1_000),
            KernelRow("unexpected_kernel", 500),
        ]
        result = summarize_kernels(rows)
        self.assertEqual(sum(item.total_ns for item in result), 6_500)
        self.assertEqual(result[0].family, "gemm_fused")
        self.assertEqual(result[-1].family, "other")
```

- [ ] **Step 2: Run the focused test and verify RED**

Run in Docker:

```bash
python3 -m unittest tests/test_stream1_transformer_profile_summary.py
```

Expected: failure because `tools.stream1_transformer_profile_summary` does not exist.

- [ ] **Step 3: Implement the parser and stable family classifier**

Define immutable `KernelRow(name: str, duration_ns: int)` and `KernelFamilySummary(family, launches, total_ns, names)`. Read SQLite using Python `sqlite3`, discover the kernel table and string table by schema inspection, resolve demangled names, classify with ordered explicit patterns, and render Markdown. Reject absent tables, negative durations, and zero matched kernel rows with clear exceptions.

- [ ] **Step 4: Run unit tests and capture the deterministic baseline**

Build the current `stream_benchmark` and run eager and CUDA Graph at `b_micro=512, concurrency=1`, 20 processes each through `tools/stream1_transformer_autotune.py --policies baseline`. Require one identical dump SHA across both modes before profiling.

- [ ] **Step 5: Capture Nsight Systems and Nsight Compute evidence**

Profile a warmed CUDA Graph workload with Nsight Systems. Use the summary tool to produce family shares. Run Nsight Compute with one representative launch from each dominant family and collect:

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
smsp__warps_active.avg.pct_of_peak_sustained_active
launch__registers_per_thread
launch__shared_mem_per_block_dynamic
lts__t_sector_hit_rate.pct
l1tex__t_sector_hit_rate.pct
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
```

If a metric is unavailable in NCU 2025.1.1, record `unsupported` in the report rather than substituting a differently named metric silently.

- [ ] **Step 6: Document and commit the baseline**

Record exact commands, GPU/driver/CUDA, hashes, timings, kernel shares, counters, and profiler limitations in `test_results/local3070_transformer_memory_profile_2026-07-19.md`. Update `memory/CHANGELOG.md`, run `git diff --check`, and commit only Task 1 source, test, report, and memory entries.

---

### Task 2: Independent compiled GEMM-family policy contract

**Files:**
- Modify: `cuda/stream1_transformer_gemm_policy.hpp`
- Modify: `tests/stream1_transformer_gemm_policy_tests.cpp`
- Modify: `cuda/stream1_transformer.cu`

**Interfaces:**
- Consumes: environment variables `BEAM_STREAM1_TRANSFORMER_QKV_POLICY`, `BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY`, `BEAM_STREAM1_TRANSFORMER_FF1_POLICY`, `BEAM_STREAM1_TRANSFORMER_FF2_POLICY`, and `BEAM_STREAM1_TRANSFORMER_CLS_POLICY`.
- Produces: independent enums and strict parsers; unset selects `baseline`, while unknown names throw `std::invalid_argument`.

- [ ] **Step 1: Add failing parser and shape-mapping tests**

For each family, test unset, `baseline`, every compiled candidate, unknown values, and reported threadblock/warp/stage metadata. Initially compile these candidate sets:

```text
QKV: baseline, m128n128, m64n128
attention output: baseline, m128n128, m64n64
FF1: baseline, m64n128, m128n128
FF2: baseline, m64n64, m128n128
CLS: baseline, m64n64, m32n64
```

Tests must prove that selecting one family does not change metadata returned for another family.

- [ ] **Step 2: Run the focused C++ test and verify RED**

```bash
cmake --build build-fused-ln-local --target stream1_transformer_gemm_policy_tests -j2
ctest --test-dir build-fused-ln-local -R stream1_transformer_gemm_policy_tests --output-on-failure
```

Expected: compile failure because the new family parsers and metadata types are absent.

- [ ] **Step 3: Implement a single typed policy descriptor contract**

Add `Stream1TransformerGemmFamily`, `Stream1TransformerGemmPolicy`, and `Stream1TransformerGemmPolicyDesc {threadblock_m, threadblock_n, threadblock_k, warp_m, warp_n, warp_k, stages}`. Implement family-specific allowed-policy validation and names without accepting a policy merely because another family compiles it.

- [ ] **Step 4: Instantiate and dispatch family-specific CUTLASS kernels**

Template QKV, attention-output/residual, FF1, FF2, and final-CLS launchers on their threadblock, warp, and stage shapes. Parse each family variable once per launch path. Preserve baseline instantiations and the exact epilogue/rounding order. Do not add a generic runtime fallback.

- [ ] **Step 5: Build and pass focused/full regression tests**

Build `stream_benchmark`, `stream1_transformer_cuda_tests`, `stream1_transformer_gemm_policy_tests`, and `production_runner`; run full CTest. Expected: all 15 existing tests plus the expanded policy tests pass.

- [ ] **Step 6: Run racecheck on baseline and one non-baseline mixed policy**

Run Compute Sanitizer racecheck for baseline and one configuration with exactly one family changed. Expected: `0 errors, 0 warnings` for both.

- [ ] **Step 7: Update evidence and commit**

Document compiled policies, build size impact, focused/full test results, and racecheck logs. Update `memory/CHANGELOG.md`, run `git diff --check`, and commit Task 2 files only.

---

### Task 3: Multi-family exact-output autotuner

**Files:**
- Modify: `tools/stream1_transformer_autotune.py`
- Modify: `tests/test_stream1_transformer_autotune.py`
- Create: `tools/stream1_transformer_policy_from_cache.py`

**Interfaces:**
- Consumes: candidate map `dict[str, list[str]]`, fixed benchmark signature, and full binary score dumps.
- Produces: JSON cache containing independent `selected_policies`, per-family baseline/candidate medians, rejection reasons, and exact evidence; the cache launcher emits validated environment assignments.

- [ ] **Step 1: Add failing coordinate-descent selection tests**

Synthetic observations must prove that the tuner changes one family at a time, compares every candidate dump with the same deterministic baseline bytes, rejects nondeterminism, enforces 20 repetitions and 3 percent improvement, and retains the previously selected policies while tuning the next family.

- [ ] **Step 2: Add failing cache-validation tests**

Test schema, GPU, dtype, shape list, batch, concurrency, epilogue list, model fingerprint, unknown family, unknown policy, missing family, and exact valid cache. Every invalid case must return baseline assignments plus a reason before benchmark launch; direct unknown CUDA policy values must still throw.

- [ ] **Step 3: Run Python tests and verify RED**

```bash
python3 -m unittest tests/test_stream1_transformer_autotune.py
```

Expected: failures for the absent multi-family schema and selector.

- [ ] **Step 4: Implement coordinate-descent tuning and schema version 2**

Start from all-baseline. For each family, warm and measure every candidate while holding other selected families fixed. Compare every dump byte to the original all-baseline oracle. Select only a stable candidate with at least 3 percent median gain, then carry it into the next family. Persist all rejected rows and atomic-write the cache.

- [ ] **Step 5: Implement strict cache-to-environment output**

`stream1_transformer_policy_from_cache.py` loads schema v2, checks exact signature equality and per-family compiled policy membership, then emits only the five explicit environment assignments. Any validation failure emits the five baseline assignments and a diagnostic to stderr.

- [ ] **Step 6: Pass tests and commit**

Run Python tests, C++ policy tests, `git diff --check`, and commit Task 3 files only.

---

### Task 4: RTX 3070 sweep and end-to-end selection

**Files:**
- Create: `test_results/local3070_transformer_multifamily_autotune_2026-07-19.md`
- Create: `test_results/local3070_transformer_multifamily_policy_v2.json`
- Modify: `memory/CHANGELOG.md`

**Interfaces:**
- Consumes: Tasks 1-3 binaries and tuner.
- Produces: a versioned RTX 3070 policy cache or an explicit all-baseline decision, with isolated and pipeline evidence.

- [ ] **Step 1: Run the baseline versus `m64n128` FF1 rerun**

At `b_micro=512, concurrency=1`, run eager and CUDA Graph for 20 processes per policy. Require one baseline hash and byte equality for every `m64n128` dump. This determines whether the old rejection came from the fixed LayerNorm race.

- [ ] **Step 2: Run the one-family-at-a-time CUDA Graph sweep**

Tune QKV, attention-output, FF1, FF2, and CLS in that order. Record median, median absolute deviation, full-dump SHA, status, memory, and rejection reason for every row.

- [ ] **Step 3: Validate the selected combined policy**

Run 20 fresh independent processes with the combined selection. Compare every dump byte with a fresh all-baseline oracle and run racecheck. Expected: one identical hash and zero racecheck hazards.

- [ ] **Step 4: Benchmark Stream1 and Stream1-to-2-to-3**

Run at least five warmed repetitions for baseline and selected policy with the same batch/concurrency. Accept the cache only if the targeted-stage threshold is met and the pipeline median does not regress. Otherwise write an all-baseline cache and retain the negative measurements.

- [ ] **Step 5: Run final verification and document**

Build all affected targets, run full CTest, Python tests, `git diff --check`, and verify `tools/production_runner.cu` has no unintended MLP changes. Write the final RTX 3070 report and update `memory/CHANGELOG.md`.

- [ ] **Step 6: Commit the measured selection**

Commit the cache, report, and memory entry. Do not encode the RTX 3070 selection as an A100 default.

---

## Next Plan Boundary

After Task 4, use the post-fix counter report and selected GEMM layout to write a separate implementation plan for the full 51x51 FlashAttention-2-style kernel, the Q=1 CLS kernel, and QKV output-layout fusion. That plan must select candidate Br/Bc, warp count, stages, and `cp.async` buffering from the measured register/shared-memory/occupancy limits rather than guessing them here.
