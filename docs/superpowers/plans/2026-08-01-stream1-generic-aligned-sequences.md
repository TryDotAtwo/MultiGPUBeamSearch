# Stream1 Generic Aligned Sequences Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make native CUDA/CUTLASS `piece_transformer` inference accept manifest-defined puzzle sequence lengths, including Cube4 logical length 57, by running on an aligned physical sequence with correct padding masks.

**Architecture:** Add one pure shape planner that preserves the manifest length as `logical_seq_len` and selects `padded_seq_len` from the active native backend alignment. Thread both lengths through scratch allocation, input construction, GEMM strides, and FMHA; padded rows are zeroed and padded keys are excluded from softmax. Existing exact 51-token behavior remains a regression gate, while 57-token real weights become the new acceptance case.

**Tech Stack:** C++17, CUDA C++, CUTLASS example 41 FMHA, CMake/CTest, Python/PyTorch oracle, CUDA Graphs, SLURM on A100 SM80.

## Global Constraints

- `logical_seq_len` comes from the model manifest; runtime padding never changes model semantics or the manifest.
- `padded_seq_len = align_up(logical_seq_len, sequence_alignment)` where alignment is owned by the active native attention backend.
- Do not introduce a fixed maximum such as 51, 57, or 64; retain fail-closed overflow, State128, manifest, and device-capacity checks.
- Do not add runtime allocation, CPU readback, LibTorch dispatch, or fallback inference.
- Padded keys must contribute exactly zero softmax probability; zero-filled K/V rows alone are not a valid mask.
- Padded query rows are skipped or discarded and zeroed before reuse.
- MLP, Streams 2-5, State128 padding, CandidateMeta, Stream 3 payload ids, and Stream 4 semantics remain unchanged.
- Every meaningful change updates `memory/CHANGELOG.md`, requirements are recorded in `memory/PROMPTS.md`, and verification artifacts are stored under `test_results/`.

## File Structure

- `cuda/stream1_transformer_shape.hpp`: pure checked alignment and logical/physical shape contract shared by host planning and CUDA launch code.
- `cuda/stream1.hpp`: carries logical and padded sequence lengths in native transformer dimensions.
- `tools/stream1_weight_io.hpp`: derives the physical plan, allocates padded scratch, and slices lanes with physical strides.
- `cuda/stream1_transformer.cu`: constructs/zeros aligned tokens and uses physical row counts in layer GEMMs while preserving logical semantics.
- `cuda/stream1_transformer_fmha.hpp`: changes FMHA entry points to consume the explicit sequence plan.
- `cuda/stream1_transformer_fmha.cu`: launches variable logical/padded FMHA and applies a key-validity mask.
- `tests/stream1_transformer_shape_tests.cpp`: CPU RED/GREEN tests for alignment, overflow, and unsupported backend shapes.
- `tests/contract_tests.cpp`: manifest and scratch-byte regression coverage for logical 51 and 57.
- `tests/stream1_transformer_cuda_tests.cu`: native-vs-oracle score and padded-tail correctness for real 51/57 bundles.
- `tools/stream_benchmark_transformer.cu`: reports logical/padded shape, scratch bytes, graph throughput, and digest.
- `CMakeLists.txt`: registers the new CPU shape test and keeps CUDA tests in CTest.
- `memory/PROMPTS.md`, `memory/CHANGELOG.md`, `test_results/stream1_transformer_generic_aligned_sequences_2026-08-01.md`: requirements, change history, and evidence.

---

### Task 1: Checked sequence shape planner

**Files:**
- Create: `cuda/stream1_transformer_shape.hpp`
- Create: `tests/stream1_transformer_shape_tests.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: manifest `seq_len` and a backend-owned positive alignment.
- Produces: `beam::Stream1TransformerSequencePlan { std::uint32_t logical_seq_len; std::uint32_t padded_seq_len; std::uint32_t alignment; }` and `make_stream1_transformer_sequence_plan(std::uint32_t logical_seq_len, std::uint32_t alignment)`.

- [ ] **Step 1: Write the failing CPU contract test**

```cpp
#include "stream1_transformer_shape.hpp"

const auto p51 = beam::make_stream1_transformer_sequence_plan(51U, 16U);
require(p51.logical_seq_len == 51U && p51.padded_seq_len == 64U && p51.alignment == 16U);
const auto p57 = beam::make_stream1_transformer_sequence_plan(57U, 16U);
require(p57.logical_seq_len == 57U && p57.padded_seq_len == 64U);
require_throws([] { beam::make_stream1_transformer_sequence_plan(0U, 16U); }, "logical_seq_len");
require_throws([] { beam::make_stream1_transformer_sequence_plan(57U, 0U); }, "alignment");
require_throws([] { beam::make_stream1_transformer_sequence_plan(UINT32_MAX, 16U); }, "overflow");
```

- [ ] **Step 2: Run the isolated test and record RED**

Run:

```bash
cmake -S . -B build-seq-align -DCMAKE_BUILD_TYPE=Release -DMGT_ENABLE_CUDA=ON
cmake --build build-seq-align --target stream1_transformer_shape_tests --parallel 16
```

Expected: build fails because `stream1_transformer_shape.hpp` and its API do not exist.

- [ ] **Step 3: Implement the minimal checked planner**

```cpp
namespace beam {
struct Stream1TransformerSequencePlan {
    std::uint32_t logical_seq_len;
    std::uint32_t padded_seq_len;
    std::uint32_t alignment;
};

inline Stream1TransformerSequencePlan make_stream1_transformer_sequence_plan(
    std::uint32_t logical_seq_len,
    std::uint32_t alignment) {
    if (logical_seq_len == 0U) throw std::invalid_argument("logical_seq_len must be positive");
    if (alignment == 0U) throw std::invalid_argument("sequence alignment must be positive");
    const std::uint64_t padded =
        ((static_cast<std::uint64_t>(logical_seq_len) + alignment - 1U) / alignment) * alignment;
    if (padded > UINT32_MAX) throw std::overflow_error("padded_seq_len overflow");
    return {logical_seq_len, static_cast<std::uint32_t>(padded), alignment};
}
} // namespace beam
```

- [ ] **Step 4: Register and run GREEN**

Run:

```bash
cmake --build build-seq-align --target stream1_transformer_shape_tests --parallel 16
ctest --test-dir build-seq-align -R '^stream1_transformer_shape_tests$' --output-on-failure
```

Expected: PASS for 51→64, 57→64, invalid zeroes, and overflow.

- [ ] **Step 5: Commit the planner**

```bash
git add cuda/stream1_transformer_shape.hpp tests/stream1_transformer_shape_tests.cpp CMakeLists.txt
git commit -m "test: define aligned transformer sequence contract"
```

### Task 2: Padded scratch planning and lane slicing

**Files:**
- Modify: `cuda/stream1.hpp`
- Modify: `tools/stream1_weight_io.hpp`
- Modify: `tests/contract_tests.cpp`

**Interfaces:**
- Consumes: `make_stream1_transformer_sequence_plan(model.seq_len, 16U)` for the current CUTLASS tensor-attention backend.
- Produces: `Stream1TransformerDims::logical_seq_len`, `Stream1TransformerDims::padded_seq_len`, and scratch offsets based only on `padded_seq_len`.

- [ ] **Step 1: Add failing 51/57 scratch assertions**

```cpp
Stream1ModelConfig p900 = transformer_model_fixture(51U, 50U);
Stream1ModelConfig cube4 = transformer_model_fixture(57U, 56U);
const auto p900_plan = transformer_scratch_byte_plan(p900, 8U);
const auto cube4_plan = transformer_scratch_byte_plan(cube4, 8U);
require(p900_plan.logical_seq_len == 51U && p900_plan.padded_seq_len == 64U, "p900 alignment");
require(cube4_plan.logical_seq_len == 57U && cube4_plan.padded_seq_len == 64U, "cube4 alignment");
require(p900_plan.token_bytes == fp16_bytes(8ULL * 64ULL * 256ULL), "p900 padded tokens");
require(cube4_plan.qkv_bytes == fp16_bytes(8ULL * 64ULL * 3ULL * 256ULL), "cube4 padded qkv");
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cmake --build build-seq-align --target contract_tests --parallel 16
ctest --test-dir build-seq-align -R '^contract_tests$' --output-on-failure
```

Expected: compile failure because byte plans and dimensions expose no physical length, or assertion failure because allocations still use 51/57.

- [ ] **Step 3: Thread the shape plan through host allocation**

```cpp
struct TransformerScratchBytePlan {
    std::uint64_t rows = 0;
    std::uint32_t logical_seq_len = 0;
    std::uint32_t padded_seq_len = 0;
    // existing byte fields remain
};

const auto seq = make_stream1_transformer_sequence_plan(model.seq_len, 16U);
plan.logical_seq_len = seq.logical_seq_len;
plan.padded_seq_len = seq.padded_seq_len;
plan.token_bytes = fp16_bytes(rows * seq.padded_seq_len * model.d_model);
plan.qkv_bytes = fp16_bytes(rows * seq.padded_seq_len * 3ULL * model.d_model);
plan.context_bytes = fp16_bytes(rows * seq.padded_seq_len * model.d_model);
plan.ff_hidden_bytes = fp16_bytes(rows * seq.padded_seq_len * model.ff_dim);
```

Update `transformer_attention_score_stride` and every lane pointer increment to use the same checked `padded_seq_len`. Keep `model.seq_len` as the manifest/logical value.

- [ ] **Step 4: Run GREEN plus memory-accounting regressions**

Run:

```bash
cmake --build build-seq-align --target contract_tests dispatcher_cuda_tests --parallel 16
ctest --test-dir build-seq-align -R '^(contract_tests|dispatcher_cuda_tests)$' --output-on-failure
```

Expected: both pass; lane regions do not overlap and Cube4 allocation uses 64 physical rows.

- [ ] **Step 5: Commit scratch planning**

```bash
git add cuda/stream1.hpp tools/stream1_weight_io.hpp tests/contract_tests.cpp
git commit -m "feat: allocate aligned transformer sequence scratch"
```

### Task 3: Logical input construction with deterministic zero tail

**Files:**
- Modify: `cuda/stream1_transformer.cu`
- Modify: `tests/stream1_transformer_cuda_tests.cu`

**Interfaces:**
- Consumes: `network.dims.logical_seq_len` and `network.dims.padded_seq_len` from Task 2.
- Produces: `scratch.tokens[row, token, dim]` with normal CLS/piece values for `token < logical_seq_len` and bitwise zero for `logical_seq_len <= token < padded_seq_len`.

- [ ] **Step 1: Add a failing CUDA tail test for logical 57, padded 64**

```cpp
launch_stream1_transformer_input_for_test(states, view57, scratch57, rows, stream);
std::vector<half> tokens(rows * 64U * 256U);
cudaMemcpy(tokens.data(), scratch57.tokens, tokens.size() * sizeof(half), cudaMemcpyDeviceToHost);
for (std::uint32_t row = 0; row < rows; ++row)
  for (std::uint32_t token = 57U; token < 64U; ++token)
    for (std::uint32_t dim = 0; dim < 256U; ++dim)
      require(__half_as_ushort(tokens[(row * 64U + token) * 256U + dim]) == 0U, "padded token tail");
```

- [ ] **Step 2: Run and verify RED on CUDA**

Run:

```bash
cmake --build build-seq-align --target stream1_transformer_cuda_tests --parallel 16
BEAM_STREAM1_TRANSFORMER_REFERENCE_DIR=test_results/stream1_transformer_reference ./build-seq-align/stream1_transformer_cuda_tests
```

Expected: current `block51` validation rejects 57 or the physical tail is non-zero.

- [ ] **Step 3: Generalize input kernels and zero physical rows**

```cpp
const std::uint32_t physical_token = blockIdx.x % network.dims.padded_seq_len;
if (physical_token >= network.dims.logical_seq_len) {
    tokens[(static_cast<std::uint64_t>(row) * network.dims.padded_seq_len + physical_token) *
           network.dims.d_model + dim] = __float2half(0.0f);
    return;
}
// token 0 is CLS; token 1..logical_seq_len-1 maps to manifest piece metadata.
```

Replace the exact `seq=51/pieces=50/classes=120` launch guard with structural checks: `logical_seq_len == num_pieces + 1`, positive dimensions, valid `piece_positions`, and `padded_seq_len >= logical_seq_len`. Preserve p900’s specialized input only as an optional shape-selected optimization; the generic aligned kernel is the required path for every valid manifest shape.

- [ ] **Step 4: Run GREEN for zero tail and existing 51 fixture**

Run:

```bash
cmake --build build-seq-align --target stream1_transformer_cuda_tests --parallel 16
BEAM_STREAM1_TRANSFORMER_REFERENCE_DIR=test_results/stream1_transformer_reference ./build-seq-align/stream1_transformer_cuda_tests
```

Expected: padded tail is bitwise zero and the existing 51-token score tolerance still passes.

- [ ] **Step 5: Commit generic input construction**

```bash
git add cuda/stream1_transformer.cu tests/stream1_transformer_cuda_tests.cu
git commit -m "feat: build zero-padded transformer token rows"
```

### Task 4: Variable-length masked CUTLASS FMHA

**Files:**
- Modify: `cuda/stream1_transformer_fmha.hpp`
- Modify: `cuda/stream1_transformer_fmha.cu`
- Modify: `cuda/stream1_transformer.cu`
- Modify: `tests/stream1_transformer_cuda_tests.cu`

**Interfaces:**
- Consumes: QKV laid out with `padded_seq_len * 3 * d_model` per batch, plus logical and padded lengths.
- Produces: attention context for logical queries only; padded keys have zero probability and padded context rows are zero.

- [ ] **Step 1: Add a failing mask-isolation CUDA test**

Create two identical logical-57 QKV inputs with physical length 64. Fill the seven padded K/V rows in input A with zero and in input B with large finite sentinel values, invoke the masked FMHA entry point, and assert logical contexts are byte- or tolerance-equal:

```cpp
stream1_transformer_fmha_attention_cuda(qkv_a, packed_a, out_a, dims57x64, false, batch, stream);
stream1_transformer_fmha_attention_cuda(qkv_b, packed_b, out_b, dims57x64, false, batch, stream);
require(max_abs_diff(out_a, out_b, batch * 57U * 256U) <= fp16_mask_tolerance,
        "padded keys changed logical attention");
require(device_tail_is_zero(out_a, batch, 57U, 64U, 256U), "padded context tail");
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cmake --build build-seq-align --target stream1_transformer_cuda_tests --parallel 16
./build-seq-align/stream1_transformer_cuda_tests
```

Expected: current FMHA rejects `seq_len=57`, or sentinels alter logical output because `NoCustomMask` includes padded keys.

- [ ] **Step 3: Pass variable strides and a key mask into CUTLASS**

Change the FMHA launch interface to use explicit shape fields:

```cpp
void stream1_transformer_fmha_attention_cuda(
    half* qkv, half* packed_qkv, half* context,
    Stream1TransformerDims dims, bool sm75_fp16,
    std::uint32_t b_micro, cudaStream_t stream);
```

Inside the typed launcher set:

```cpp
params.num_queries = static_cast<int32_t>(dims.logical_seq_len);
params.num_keys = static_cast<int32_t>(dims.padded_seq_len);
params.q_strideM = params.k_strideM = params.v_strideM = 3 * dims.d_model;
params.q_strideB = params.k_strideB = params.v_strideB =
    static_cast<int64_t>(dims.padded_seq_len) * 3 * dims.d_model;
```

Use the CUTLASS example-41 custom mask facility available in the pinned headers so positions `key >= logical_seq_len` receive negative infinity before softmax. If that header exposes only compile-time mask modes, pre-pack QKV plus a preallocated validity/bias region in `attention_scores_probs` and dispatch the existing native masked attention kernel; do not allocate or read back at launch time. Zero `[logical_seq_len, padded_seq_len)` in `context` before returning.

- [ ] **Step 4: Run masked 57 and 51 regressions**

Run:

```bash
cmake --build build-seq-align --target stream1_transformer_cuda_tests --parallel 16
ctest --test-dir build-seq-align -R '^stream1_transformer_cuda_tests$' --output-on-failure
```

Expected: sentinel isolation passes, physical context tail is zero, and the 51-token oracle remains within score-key tolerance.

- [ ] **Step 5: Commit masked attention**

```bash
git add cuda/stream1_transformer_fmha.hpp cuda/stream1_transformer_fmha.cu cuda/stream1_transformer.cu tests/stream1_transformer_cuda_tests.cu
git commit -m "feat: mask aligned transformer attention padding"
```

### Task 5: Real Cube4 oracle and CUDA Graph acceptance

**Files:**
- Modify: `tests/stream1_transformer_cuda_tests.cu`
- Modify: `tests/test_stream1_transformer_parity.py`
- Modify: `tools/stream_benchmark_transformer.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: exported real-weight bundles selected by `BEAM_STREAM1_TRANSFORMER_REFERENCE_DIR` (51) and `BEAM_STREAM1_TRANSFORMER_CUBE4_REFERENCE_DIR` (57).
- Produces: score-key parity, score digest, graph replay throughput, scratch byte count, and shape fields in machine-readable output.

- [ ] **Step 1: Generate the Cube4 PyTorch reference artifact**

Run the existing exporter/oracle against the downloaded model bundle and store only deterministic states, FP32 oracle scores, manifest, and file digests under `test_results/stream1_transformer_cube4_reference/`. The parity test must assert:

```python
assert manifest["seq_len"] == 57
assert manifest["num_pieces"] == 56
assert manifest["d_model"] == 256
assert manifest["nhead"] == 8
assert manifest["head_dim"] == 32
```

- [ ] **Step 2: Observe the real-weight RED before enabling the new route**

Run:

```bash
BEAM_STREAM1_TRANSFORMER_CUBE4_REFERENCE_DIR=test_results/stream1_transformer_cube4_reference \
  ./build-seq-align/stream1_transformer_cuda_tests
```

Expected before Tasks 2-4: `seq_len=51` rejection. After Tasks 2-4 this becomes the acceptance check and must pass.

- [ ] **Step 3: Add graph replay and reporting fields**

Capture the full 57-token inference into a CUDA Graph, replay at least 100 warm iterations and 1000 timed iterations, and print:

```text
logical_seq_len=57 padded_seq_len=64 sequence_alignment=16 scratch_bytes=<bytes>
graph_replays=1000 parents_per_second=<value> score_digest=<sha256>
```

The test must fail on any `cudaMalloc`, host synchronization, or CPU score readback inside the captured steady-state region.

- [ ] **Step 4: Run complete local correctness gates**

Run:

```bash
cmake --build build-seq-align --parallel 16
ctest --test-dir build-seq-align -R 'stream1_transformer|contract_tests|dispatcher_cuda_tests' --output-on-failure
python -m pytest tests/test_stream1_transformer_parity.py tests/test_stream1_transformer_exporter.py -q
```

Expected: all pass; both 51 and 57 score keys are within the established `3072` tolerance, score digests are recorded, and graph capture/replay succeeds.

- [ ] **Step 5: Commit real-weight and graph gates**

```bash
git add tests/stream1_transformer_cuda_tests.cu tests/test_stream1_transformer_parity.py tools/stream_benchmark_transformer.cu CMakeLists.txt test_results/stream1_transformer_cube4_reference
git commit -m "test: gate Cube4 native transformer parity"
```

### Task 6: A100 performance gate, documentation, and cluster handoff

**Files:**
- Modify: `memory/PROMPTS.md`
- Modify: `memory/CHANGELOG.md`
- Create: `test_results/stream1_transformer_generic_aligned_sequences_2026-08-01.md`

**Interfaces:**
- Consumes: exact commit from Tasks 1-5 and cluster model/data paths already established under `/mnt/pool/6/vokirova/beam444a100`.
- Produces: validated A100 evidence and one full pasteable production submission command pinned to the tested commit.

- [ ] **Step 1: Run the 51/57 A100 benchmark in one SLURM job**

Use one complete `set -euo pipefail` command that checks out the exact commit, builds SM80 Release targets, runs CTest, benchmarks logical 51 and 57 with CUDA Graphs, and prints `JOB_ID`, `OUTPUT`, final `sacct`, and the entire log. Record alignment, scratch bytes, throughput, and digest for both shapes.

- [ ] **Step 2: Enforce the performance acceptance rule**

Compare 51-token graph throughput against the pre-change A100 baseline using identical batch, dtype, policies, concurrency, warmup, and iteration count. Acceptance requires no material regression; treat `>3%` slowdown as material and investigate before production. Confirm 57 has no steady-state allocation and produces a stable digest across repeated graph runs.

- [ ] **Step 3: Record requirements, changes, and evidence**

Append to `memory/PROMPTS.md` the user requirement: generic aligned/masked native inference for every supported puzzle, nearest backend alignment, zero padding, no LibTorch. Append implementation and commit references to `memory/CHANGELOG.md`. Write exact commands, GPU/driver, job id, timings, scratch sizes, digests, and test outputs to `test_results/stream1_transformer_generic_aligned_sequences_2026-08-01.md`.

- [ ] **Step 4: Run final repository gates**

Run:

```bash
git diff --check
git status --short
ctest --test-dir build-seq-align --output-on-failure
python -m pytest tests/test_stream1_transformer_parity.py tests/test_stream1_transformer_exporter.py -q
```

Expected: clean checks, only intended scoped files, all tests pass, and evidence references the exact tested commit/job.

- [ ] **Step 5: Commit and push the scoped branch**

```bash
git add memory/PROMPTS.md memory/CHANGELOG.md test_results/stream1_transformer_generic_aligned_sequences_2026-08-01.md
git commit -m "docs: record aligned transformer A100 validation"
git push -u origin codex/stream1-generic-seq-align
```

- [ ] **Step 6: Produce the 400M search command**

Give the user one pasteable cluster block in their established style. It must pin and verify the full tested commit, use `/mnt/pool/6/vokirova/beam444a100`, submit via `sbatch --parsable -p kaf12`, request 8 GPUs and 128G RAM, use the real Cube4 bundle and native transformer weights, launch beam 400M, wait through `squeue`, then print `sacct` and `cat` the complete job log.

## Self-Review

- Spec coverage: Tasks 1-4 cover logical/physical shape, alignment, allocation, zero tail, masking, no fallback, and generic dispatch. Task 5 covers 51/57 parity and CUDA Graphs. Task 6 covers A100 performance, documentation, scoped push, and the production command.
- Placeholder scan: no `TBD`, `TODO`, “implement later”, or undefined follow-up task remains.
- Type consistency: `Stream1TransformerSequencePlan`, `logical_seq_len`, `padded_seq_len`, and `alignment` use the same names and unsigned 32-bit types in every task; scratch byte counts use unsigned 64-bit types.
