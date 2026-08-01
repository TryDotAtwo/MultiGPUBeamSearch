# Stream1 Piece-Transformer Production Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Stream1 `piece_transformer` backend measurable and optimizable toward the same production quality bar as the existing MLP backend, while preserving MLP behavior.

**Architecture:** Keep `STREAM1_BACKEND_MLP` and `STREAM1_BACKEND_PIECE_TRANSFORMER` as explicit separate runtime paths. The first implementation step is a transformer-only microbenchmark/profiling path; subsequent speed work must be driven by those measurements and must not alter Stream3/Stream4 contracts or MLP execution semantics.

**Tech Stack:** CUDA C++17, CUTLASS GEMM wrappers already in `cuda/stream1.cu`, CMake targets, existing `stream1_weights` loader/exporter contracts, Kaggle 2xT4 and Docker GPU validation.

---

## File Structure

- `tools/stream_benchmark.cu`: add a transformer-only Stream1 benchmark path. It may share setup/helpers with the existing benchmark but must keep the current MLP benchmark output intact.
- `cuda/stream1.cu`: future hot-path changes only after Task 1 measurements. MLP functions must remain semantically unchanged.
- `tools/stream1_weight_io.hpp`: future scratch sizing changes must be backend-gated and exact; no MLP scratch layout changes.
- `cuda/runtime_config.cpp`: future transformer-specific config knobs must default to current MLP behavior and only apply when `stream1_model.backend == STREAM1_BACKEND_PIECE_TRANSFORMER`.
- `tests/dispatcher_cuda_tests.cu`, `tests/contract_tests.cpp`, `tests/stream1_transformer_cuda_tests.cu`: add focused contract coverage when behavior changes.
- `test_results/`: store benchmark and verification notes.
- `memory/CHANGELOG.md`, `memory/PROMPTS.md`: update after meaningful code/config/architecture changes per `AGENTS.md`.

---

### Task 1: Add Transformer Stream1 Microbenchmark

**Files:**
- Modify: `tools/stream_benchmark.cu`
- Test/Report: `test_results/stream1_transformer_benchmark_2026-06-30.md`

- [ ] **Step 1: Preserve the existing MLP benchmark path**

Locate the current guard:

```cpp
if (stream1_model.backend == STREAM1_BACKEND_PIECE_TRANSFORMER) {
    throw std::runtime_error("stream_benchmark is MLP-only; piece_transformer benchmark path is not implemented");
}
```

Replace it with explicit branching:

```cpp
if (stream1_model.backend == STREAM1_BACKEND_PIECE_TRANSFORMER) {
    benchmark_stream1_transformer(...);
    return 0;
}
```

The MLP path after this branch must keep calling `benchmark_stream1(...)`, `benchmark_stream2(...)`, `benchmark_stream3(...)`, and `benchmark_stream4(...)` exactly as before.

- [ ] **Step 2: Implement `benchmark_stream1_transformer`**

Create a helper in `tools/stream_benchmark.cu` that:

```cpp
std::vector<TransformerStream1BenchmarkResult> benchmark_stream1_transformer(
    const stream1_weights::DeviceWeights& weights,
    const Stream1ModelConfig& stream1_model,
    const State128* d_states,
    std::uint32_t max_states,
    std::ofstream& report);
```

Required behavior:

- Build `stream1_weights::TransformerNetworkViewHolder` from `weights.transformer`.
- Allocate transformer scratch using `stream1_weights::alloc_stream1_scratch(stream1_model, b_micro, concurrency)`.
- Allocate `parent_base`, `count`, and `score_ring` device buffers.
- Time `stream1_transformer_inference_cuda(...)` with CUDA events.
- Sweep at least `B_MICRO={512,1024,2048,4096,8192}` and `STREAM1_CONCURRENCY={1,2,4}` where memory allows.
- Skip configs whose scratch allocation would exceed available memory; report them as skipped, not failed.
- Emit rows with `b_micro`, `concurrency`, `rows_per_launch_group`, `ms_per_launch_group`, `parents_per_sec`, `candidates_per_sec`, and `scratch_bytes`.

- [ ] **Step 3: Keep the benchmark deterministic**

Use the existing `make_state_batch(...)` and `load_initial_state_from_test_csv(...)` setup. Do not add random state generation or Python dependencies.

- [ ] **Step 4: Verify locally in Docker when possible**

Run:

```bash
cmake --build <existing-or-new-build-dir> --target stream_benchmark -j 2
BEAM_WEIGHT_DIR=<piece-transformer-export-dir> BEAM_STREAM_MICRO_ONLY=1 BEAM_STREAM_BENCH_REPORT=test_results/stream1_transformer_benchmark_2026-06-30.md <build-dir>/stream_benchmark 991
```

Expected:

- exits 0 with transformer weights;
- report contains a `Stream1 Piece Transformer` table;
- existing MLP benchmark still builds and its path remains available.

- [ ] **Step 5: Update project memory and commit**

Update `memory/CHANGELOG.md` and `memory/PROMPTS.md` for this task and commit only the relevant files.

---

### Task 2: Add Transformer-Specific Runtime Knobs Without Changing MLP Defaults

**Files:**
- Modify: `cuda/runtime_config.cpp`
- Modify: `tools/production_runner.cu` only for logging if needed
- Test: `tests/dispatcher_cuda_tests.cu` or `tests/contract_tests.cpp`
- Report: `test_results/stream1_transformer_runtime_knobs_2026-06-30.md`

- [ ] **Step 1: Add backend-gated env overrides**

In `build_runtime_config_from_budget(...)`, keep MLP behavior unchanged. For `STREAM1_BACKEND_PIECE_TRANSFORMER`, allow these optional env vars:

```cpp
BEAM_TRANSFORMER_B_MICRO
BEAM_TRANSFORMER_STREAM1_CONCURRENCY
BEAM_TRANSFORMER_STREAM3_RING_SLOTS
```

Resolution order:

1. transformer-specific env var when backend is `piece_transformer`;
2. existing generic env var;
3. existing default.

- [ ] **Step 2: Add clear logs**

`production_runner` must print resolved values with current existing fields plus, for transformer backend, enough context to see whether transformer-specific overrides were used.

- [ ] **Step 3: Add contract coverage**

Add a small test that verifies:

- MLP model ignores transformer-specific env vars and keeps generic/default values;
- transformer model uses transformer-specific env vars when present;
- invalid `STREAM1_CONCURRENCY > STREAM3_RING_SLOTS` guard still fires.

- [ ] **Step 4: Verify**

Run focused tests that cover runtime config and existing dispatcher branch. Existing MLP tests must not regress.

---

### Task 3: Remove Attention Probability Global Scratch From Transformer Forward

**Files:**
- Modify: `cuda/stream1.cu`
- Modify: `tools/stream1_weight_io.hpp`
- Test: `tests/stream1_transformer_cuda_tests.cu`, `tests/dispatcher_cuda_tests.cu`
- Report: `test_results/stream1_transformer_attention_fusion_2026-06-30.md`

- [ ] **Step 1: Write a failing size/behavior test first**

Add or update a test to verify transformer scratch sizing no longer includes `rows * nhead * seq_len * seq_len * sizeof(float)` global score/probability storage after the new kernel lands.

- [ ] **Step 2: Replace global-probability attention use**

Change `stream1_transformer_attention51_kernel` so it computes score, softmax, and value accumulation using only shared/register storage per query/head. It must not write `scores_probs` to global memory.

- [ ] **Step 3: Keep correctness tolerance**

`tests/stream1_transformer_cuda_tests.cu` must still pass against the reference fixture when present. Clean checkout skip behavior must remain unchanged.

- [ ] **Step 4: Update scratch allocation**

Remove or reduce `transformer_attention_scores_probs` only for transformer backend. MLP scratch allocation must remain byte-for-byte equivalent.

- [ ] **Step 5: Benchmark before/after**

Use Task 1 benchmark report to compare rows/sec before and after this change.

---

### Task 4: Transformer Architecture Memo For Better Inference

**Files:**
- Create: `docs/transformer_inference_architecture.md`
- Report: `test_results/stream1_transformer_architecture_memo_2026-06-30.md`

- [ ] **Step 1: Document current architecture cost**

Summarize the current p900 architecture:

- state_len=120;
- num_pieces=50;
- seq_len=51;
- d_model=256;
- nhead=8;
- layers=4;
- ff_dim=1024;
- output_dim=24;
- cls pooling;
- SiLU FFN.

- [ ] **Step 2: Propose inference-friendly alternatives**

Compare at least these options:

1. Current piece transformer with optimized fused CUDA path.
2. Smaller piece transformer: fewer layers, lower `d_model`, lower `ff_dim`.
3. Piece-mixer/MLP-Mixer style model over pieces: token mixing + channel MLP without QKV attention.
4. Smaller/lower-rank transformer variants that preserve explicit transformer backend semantics.
5. Transformer reranking only as a future explicit transformer stage, not as a fallback path.

Do not recommend distilling into MLP or any fallback route. For each option, state expected inference cost, compatibility with current exporter/runtime, and risk to solution quality.

- [ ] **Step 3: Recommend a transformer-only next model family**

Recommend one transformer or transformer-like architecture to train next for beam-search inference throughput, not pure notebook convenience. MLP distillation and fallback paths are out of scope by user direction.

---

## Global Acceptance Gates

- No fallback backend behavior is introduced.
- MLP manifests, MLP runtime config, and MLP benchmark behavior remain compatible.
- Stream3/Stream4 contracts from `AGENTS.md` remain untouched unless explicitly planned later.
- All dynamic allocations for repeated inference remain outside the steady-state loop.
- Every implementation task gets spec review and code quality review before proceeding.
