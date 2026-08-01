# Local RTX 3070 Transformer GEMM Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and measure explicit SM80 FF1 CUTLASS policy candidates while preserving exact native transformer outputs and the existing default path.

**Architecture:** Keep policy selection inside the native transformer CUDA backend and default to the current `128x64x32 / 64x32x32` policy. Add two opt-in FF1 candidates selected by one fail-closed environment variable, then keep only a stable local winner. The MLP path, attention, residual GEMMs, and Stream 4 remain unchanged.

**Tech Stack:** C++17, CUDA 12.8, CUTLASS TensorOp GEMM, CMake/CTest, Docker, Nsight Systems, RTX 3070 SM86.

## Global Constraints

- Develop and profile in `gpu-dev-cutlass-nsight:2026-05-24` on the local RTX 3070.
- Use real FP16 p900 weights with `block51=1`, `final_cls_only=1`, and `final_cls_attention=0`.
- Preserve checksum `841858064`, digest `821400116975659197`, and first score keys.
- No silent fallback and no MLP or Stream 4 changes.
- Compare at least five warmed repetitions and treat less than 3% as noise.

---

### Task 1: Fail-closed FF1 policy contract

**Files:**
- Modify: `tests/stream1_transformer_cuda_tests.cu`
- Modify: `cuda/stream1_transformer.cu`

**Interfaces:**
- Consumes: environment variable `BEAM_STREAM1_TRANSFORMER_FF1_POLICY`.
- Produces: accepted values `baseline`, `m64n128`, `m64n64`; unset means `baseline`; every other value throws `std::invalid_argument`.

- [ ] **Step 1: Write the failing test**

Add a scoped environment setter and, before the normal inference call, set:

```cpp
setenv("BEAM_STREAM1_TRANSFORMER_FF1_POLICY", "invalid-policy", 1);
bool invalid_policy_rejected = false;
try {
    stream1_transformer_inference_cuda(/* existing fixture arguments */);
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
} catch (const std::invalid_argument&) {
    invalid_policy_rejected = true;
}
unsetenv("BEAM_STREAM1_TRANSFORMER_FF1_POLICY");
require(invalid_policy_rejected, "invalid transformer FF1 policy must fail closed");
```

Reset the score buffer before the existing reference inference.

- [ ] **Step 2: Verify RED**

Run in Docker:

```bash
cmake --build build-fused-ln-local --target stream1_transformer_cuda_tests -j2
./build-fused-ln-local/stream1_transformer_cuda_tests
```

Expected: FAIL with `invalid transformer FF1 policy must fail closed` because the variable is currently ignored.

- [ ] **Step 3: Implement minimal parsing**

Add:

```cpp
enum class Stream1TransformerFf1Policy { Baseline, M64N128, M64N64 };

Stream1TransformerFf1Policy stream1_transformer_ff1_policy() {
    const char* value = std::getenv("BEAM_STREAM1_TRANSFORMER_FF1_POLICY");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "baseline") == 0) {
        return Stream1TransformerFf1Policy::Baseline;
    }
    if (std::strcmp(value, "m64n128") == 0) return Stream1TransformerFf1Policy::M64N128;
    if (std::strcmp(value, "m64n64") == 0) return Stream1TransformerFf1Policy::M64N64;
    throw std::invalid_argument("unknown BEAM_STREAM1_TRANSFORMER_FF1_POLICY");
}
```

Call the parser at the start of the FP16 SM80 FF1 dispatch without changing the baseline instantiation yet.

- [ ] **Step 4: Verify GREEN**

Run the same CUDA test. Expected: `stream1_transformer_cuda_tests=pass`.

- [ ] **Step 5: Commit**

Commit only the test and parser as `Add fail-closed transformer FF1 policy contract`.

### Task 2: SM80 FF1 candidate kernels

**Files:**
- Modify: `cuda/stream1_transformer.cu`
- Modify: `tests/stream1_transformer_cuda_tests.cu`

**Interfaces:**
- Consumes: the Task 1 policy enum.
- Produces: three exact-output SM80 FP16 FF1 implementations.

- [ ] **Step 1: Write the failing candidate parity test**

Refactor the existing fixture inference into a helper returning score keys.
Run `baseline`, `m64n128`, and `m64n64`, then require exact vector equality:

```cpp
require(candidate_scores == baseline_scores,
        "transformer FF1 policy changed native score keys");
```

Expected RED: candidate policies currently dispatch the baseline and the test must additionally require a distinct reported effective policy string from a new `stream1_transformer_ff1_policy_name()` API, which is not yet defined.

- [ ] **Step 2: Verify RED**

Build the CUDA test. Expected: compile failure for the missing policy-name API.

- [ ] **Step 3: Generalize the FF1 typed wrapper**

Change the FF1 template to accept threadblock and warp shapes:

```cpp
template <typename Element, typename ArchTag, typename InstructionShape,
          typename ThreadblockShape, typename WarpShape>
void stream1_transformer_ff1_linear_bias_silu_typed(...)
```

Instantiate:

- `baseline`: `GemmShape<128,64,32>`, `GemmShape<64,32,32>`;
- `m64n128`: `GemmShape<64,128,32>`, `GemmShape<32,64,32>`;
- `m64n64`: `GemmShape<64,64,32>`, `GemmShape<32,32,32>`.

SM75 and BF16 remain on the baseline policy. If a non-baseline policy is requested outside FP16 SM80+, throw instead of falling back.

- [ ] **Step 4: Add policy-name API and exact parity execution**

Expose the effective name for tests and benchmark logging. Run all three policies against the same fixture and compare exact score vectors.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
cmake --build build-fused-ln-local --target stream1_transformer_cuda_tests stream_benchmark -j2
./build-fused-ln-local/stream1_transformer_cuda_tests
```

Expected: test passes and all policy score vectors are identical.

- [ ] **Step 6: Commit**

Commit as `Add SM80 transformer FF1 GEMM policy candidates`.

### Task 3: Stable local policy sweep

**Files:**
- Modify: `tools/stream_benchmark_transformer.cu`
- Create: `test_results/local3070_transformer_ff1_policy_sweep_2026-07-18.md`

**Interfaces:**
- Consumes: named policies from Task 2.
- Produces: five-run median and correctness evidence for each candidate.

- [ ] **Step 1: Log the effective FF1 policy**

Add `stream1_transformer_ff1_policy=<name>` to the report and stdout before timing.

- [ ] **Step 2: Run five warmed repetitions per policy**

For each policy use the fixed baseline flags and unique report files. Capture
`nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw` before
and after each run.

- [ ] **Step 3: Reject mathematically different rows**

Require every row to match checksum `841858064`, digest
`821400116975659197`, and the baseline first-score-key sequence before computing
the median.

- [ ] **Step 4: Select or reject the candidate**

Keep a candidate only if median throughput is at least 3% above baseline and
the result is not driven by one outlier. Prefer at least 5%. Remove losing
instantiations and their selector values so production code retains no dead
experiments.

- [ ] **Step 5: Commit the accepted result**

Commit source, tests, benchmark logging, report, `memory/CHANGELOG.md`, and
`memory/PROMPTS.md` as one evidence-backed optimization commit. If neither
candidate wins, commit only the fail-closed measurement tooling and rejection
report.

### Task 4: Pipeline verification and re-profile

**Files:**
- Create: `test_results/local3070_transformer_ff1_policy_pipeline_2026-07-18.md`
- Modify: `memory/CHANGELOG.md`

**Interfaces:**
- Consumes: Task 3 winner or baseline if both candidates lose.
- Produces: final isolated/pipeline decision and the next measured bottleneck.

- [ ] **Step 1: Run the complete focused test set**

Run `ctest --test-dir build-fused-ln-local --output-on-failure` and require zero failures.

- [ ] **Step 2: Run five Stream1-3 pipeline repetitions**

Use `window=32`, `b_micro=512`, `concurrency=2`, `ring_slots=8`, and
`stream3_batch=98304`. Compare medians against the recorded baseline and reject
more than 2% regression.

- [ ] **Step 3: Re-profile the selected path**

Run bounded Nsight Systems with CUDA graph node tracing and record the updated
kernel-family shares. Use NCU only on the new largest individual kernel family.

- [ ] **Step 4: Record the next decision**

If FF1 remains dominant, use its NCU counters for another small policy change.
Otherwise move to residual GEMM tuning. Do not begin final `Q_cls + KV_all`
until the per-operation GEMM stage has a documented stopping point.

- [ ] **Step 5: Commit verification evidence**

Commit the pipeline report and changelog update as `Verify local transformer FF1 policy winner`.
