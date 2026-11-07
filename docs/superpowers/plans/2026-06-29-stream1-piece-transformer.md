# Stream1 Piece Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit `piece_transformer` Stream1 backend so the existing beam search can run the Kaggle Megaminx Q-transformer model with `output_dim == MOVE_COUNT` on small 2xT4 smoke runs first, without breaking the current MLP backend.

**Architecture:** Keep the MLP Stream1 path intact and add a separate transformer backend selected from `manifest.json`. The transformer path exports Kaggle's fast-form input embedding (`fast_slot_projected` and `fast_piece_static`), runs standalone CUDA correctness tests against a PyTorch reference, then wires the backend into dispatcher graph capture only after the standalone tests pass.

**Tech Stack:** C++20, CUDA 17, CUTLASS GEMM path, Python/PyTorch exporter/reference generation, Kaggle CLI for 2xT4 smoke validation.

---

## File Structure

- Modify `src/config.hpp` and `src/config.cpp`: add explicit Stream1 backend enum and transformer dimensions while preserving MLP defaults.
- Modify `tools/export_stream1_transformer.py`: keep automatic manifest generation and add reference-vector export support for tests.
- Create `tools/stream1_transformer_weight_io.hpp`: load/upload transformer weights and allocate transformer scratch independently of `tools/stream1_weight_io.hpp`.
- Create `cuda/stream1_transformer.hpp` and `cuda/stream1_transformer.cu`: standalone transformer Stream1 forward that writes `score_ring[parent * MOVE_COUNT + move]`.
- Modify `cuda/dispatcher.hpp` and `cuda/dispatcher.cu`: store a tagged Stream1 runtime and dispatch either MLP or transformer inside the existing graph capture.
- Modify `cuda/runtime_config.cpp`: estimate transformer weights/scratch separately.
- Modify `tools/production_runner.cu`: load tagged model config, print backend-specific diagnostics, allocate the correct weights/scratch, and free them safely.
- Create `tests/stream1_transformer_cuda_tests.cu`: deterministic CUDA tests using exported reference vectors.
- Modify `CMakeLists.txt`: add transformer CUDA source and test executable.
- Create `kaggle_t4_transformer_smoke/`: private Kaggle package for 2xT4 small-beam validation after local/Docker tests pass.
- Update `memory/PROMPTS.md`, `memory/CHANGELOG.md`, and `test_results/` notes for each meaningful implementation milestone.

---

### Task 1: Add Tagged Stream1 Model Config

**Files:**
- Modify: `src/config.hpp`
- Modify: `src/config.cpp`
- Test: `tests/contract_tests.cpp`

- [ ] **Step 1: Add a failing config contract test**

Add this test to `tests/contract_tests.cpp` near the other config contracts:

```cpp
void test_stream1_backend_row_modes() {
    Stream1ModelConfig mlp;
    mlp.backend = STREAM1_BACKEND_MLP;
    mlp.output_dim = STREAM1_SINGLE_SCORE_OUTPUT_DIM;
    if (!stream1_uses_child_rows(mlp) || stream1_rows_per_parent(mlp) != MOVE_COUNT) {
        throw std::runtime_error("MLP one-output backend must use child rows");
    }

    Stream1ModelConfig transformer;
    transformer.backend = STREAM1_BACKEND_PIECE_TRANSFORMER;
    transformer.output_dim = static_cast<std::uint32_t>(MOVE_COUNT);
    transformer.num_pieces = 50;
    transformer.max_piece_size = 3;
    transformer.seq_len = 51;
    transformer.d_model = 256;
    transformer.nhead = 8;
    transformer.head_dim = 32;
    transformer.transformer_layers = 4;
    transformer.ff_dim = 1024;
    if (stream1_uses_child_rows(transformer) || stream1_rows_per_parent(transformer) != 1U) {
        throw std::runtime_error("piece_transformer backend must use parent rows");
    }
}
```

Call it from `main()`:

```cpp
test_stream1_backend_row_modes();
```

- [ ] **Step 2: Run the contract test and verify it fails**

Run:

```bash
cmake --build build-gpu-dev-cutlass --target contract_tests
./build-gpu-dev-cutlass/contract_tests
```

Expected: compile failure because `STREAM1_BACKEND_PIECE_TRANSFORMER` and transformer fields do not exist.

- [ ] **Step 3: Add backend fields without changing MLP defaults**

In `src/config.hpp`, add:

```cpp
inline constexpr std::uint32_t STREAM1_BACKEND_MLP = 0;
inline constexpr std::uint32_t STREAM1_BACKEND_PIECE_TRANSFORMER = 1;
```

Extend `Stream1ModelConfig`:

```cpp
struct Stream1ModelConfig {
    std::uint32_t backend = STREAM1_BACKEND_MLP;
    std::uint32_t state_len = static_cast<std::uint32_t>(STATE_LEN);
    std::uint32_t num_classes = static_cast<std::uint32_t>(STATE_LEN);
    std::uint32_t hidden1 = 1536;
    std::uint32_t hidden2 = 512;
    std::uint32_t residual_count = 2;
    std::uint32_t output_dim = static_cast<std::uint32_t>(MOVE_COUNT);
    std::uint32_t dtype = STREAM1_DTYPE_FP16;
    std::uint32_t normalization = STREAM1_NORM_NONE;
    std::uint32_t num_pieces = 0;
    std::uint32_t max_piece_size = 0;
    std::uint32_t seq_len = 0;
    std::uint32_t d_model = 0;
    std::uint32_t nhead = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t transformer_layers = 0;
    std::uint32_t ff_dim = 0;
};
```

- [ ] **Step 4: Make row-mode helpers backend-aware**

In `src/config.cpp`:

```cpp
bool stream1_uses_child_rows(const Stream1ModelConfig& model) {
    if (model.backend == STREAM1_BACKEND_PIECE_TRANSFORMER) {
        return false;
    }
    return model.output_dim == STREAM1_SINGLE_SCORE_OUTPUT_DIM;
}
```

- [ ] **Step 5: Verify existing tests still pass**

Run:

```bash
cmake --build build-gpu-dev-cutlass --target contract_tests
./build-gpu-dev-cutlass/contract_tests
```

Expected: pass.

---

### Task 2: Parse Transformer Manifest Without Breaking MLP Loader

**Files:**
- Modify: `tools/stream1_weight_io.hpp`
- Test: `tests/contract_tests.cpp`

- [ ] **Step 1: Add backend name helper**

Add to `tools/stream1_weight_io.hpp`:

```cpp
inline const char* stream1_backend_name(const Stream1ModelConfig& model) {
    return model.backend == STREAM1_BACKEND_PIECE_TRANSFORMER ? "piece_transformer" : "mlp";
}
```

- [ ] **Step 2: Extend `load_stream1_manifest()` parsing**

Parse `backend`, defaulting to `mlp`. For `piece_transformer`, parse `num_pieces`, `max_piece_size`, `seq_len`, `d_model`, `nhead`, `head_dim`, `num_layers`, and `ff_dim`. For `mlp`, keep the current hidden/residual fields exactly as they are.

- [ ] **Step 3: Split validation by backend**

MLP validation must remain current behavior. Transformer validation must require `state_len == STATE_LEN`, `output_dim == MOVE_COUNT`, `seq_len == num_pieces + 1`, `d_model % nhead == 0`, `head_dim == d_model / nhead`, `num_pieces > 0`, `max_piece_size > 0`, `transformer_layers > 0`, and `ff_dim > 0`.

- [ ] **Step 4: Verify MLP manifest still loads**

Run:

```bash
cmake --build build-gpu-dev-cutlass --target stream_benchmark
BEAM_WEIGHT_DIR=stream1_weights ./build-gpu-dev-cutlass/stream_benchmark
```

Expected: existing MLP benchmark starts and prints `stream1_model_output_dim`.

---

### Task 3: Export Deterministic Transformer Reference Vectors

**Files:**
- Modify: `tools/export_stream1_transformer.py`
- Create: `test_results/stream1_transformer_reference/README.md`
- Test: generated reference under `test_results/stream1_transformer_reference/`

- [ ] **Step 1: Add exporter arguments**

Add `--reference-out`, `--reference-count`, and `--reference-seed` to `tools/export_stream1.py` and `tools/export_stream1_transformer.py`.

- [ ] **Step 2: Add reference generation after writing weights**

When `reference_out` is set, instantiate the Kaggle model from metadata, load the checkpoint, enable fast inference, generate deterministic permutation states, and write JSON with `states`, `scores_fp32`, and metadata. The states must be generated with `torch.randperm(state_len, generator=g)` so they stay valid Megaminx-like permutations.

- [ ] **Step 3: Export fp16 T4 weights and reference locally**

Run:

```bash
python tools/export_stream1.py \
  --weights C:/tmp/megaminx_qtransformer/megaminx-transformer/weights/p900-t000-q-rw-sym_1782210824_best.pth \
  --out test_results/stream1_transformer_reference/weights_fp16 \
  --dtype fp16 \
  --format piece-transformer \
  --metadata C:/tmp/megaminx_qtransformer/megaminx-transformer/logs/model_p900-t000-q-rw-sym_1782210824.json \
  --generators C:/tmp/megaminx_qtransformer/megaminx-transformer/generators/p900.json \
  --source-root C:/tmp/megaminx_qtransformer/megaminx-transformer \
  --reference-out test_results/stream1_transformer_reference/reference.json \
  --reference-count 32
```

Expected: `manifest.json` says `backend=piece_transformer`, `output_dim=24`, `seq_len=51`.

---

### Task 4: Add Transformer Weight Loader And Scratch Arena

**Files:**
- Create: `tools/stream1_transformer_weight_io.hpp`
- Modify: `tools/production_runner.cu`
- Test: `tests/stream1_transformer_cuda_tests.cu`

- [ ] **Step 1: Create host/device structs**

Create independent transformer host/device structs. Do not add transformer tensors to the MLP `HostWeightBytes` struct. Required groups are fast input tensors, input/output LayerNorm tensors, per-block attention/FFN tensors, output head tensors, and small integer piece metadata.

- [ ] **Step 2: Load exact file sizes from manifest**

Required sizes:

```text
fast_slot_projected = max_piece_size * num_classes * d_model
fast_piece_static = num_pieces * d_model
cls_token = d_model
qkv_weight = d_model * 3*d_model
ff1_weight = d_model * ff_dim
ff2_weight = ff_dim * d_model
output_weight = d_model * output_dim
piece_positions = num_pieces * max_piece_size * sizeof(int16_t)
piece_mask = num_pieces * max_piece_size * sizeof(uint8_t)
```

- [ ] **Step 3: Allocate correctness-first scratch per lane**

Allocate `tokens_a`, `tokens_b`, `qkv`, `attn`, `ff_hidden`, and `scores`. This is the correctness form for small `BEAM_B_MICRO`; later optimization can reuse buffers more aggressively.

---

### Task 5: Implement Standalone Transformer Forward

**Files:**
- Create: `cuda/stream1_transformer.hpp`
- Create: `cuda/stream1_transformer.cu`
- Modify: `CMakeLists.txt`
- Test: `tests/stream1_transformer_cuda_tests.cu`

- [ ] **Step 1: Add public CUDA entrypoint**

Expose `stream1_transformer_inference_cuda(...)` taking current frontier states, parent base/count, transformer view/scratch, score ring, `b_micro`, and CUDA stream.

- [ ] **Step 2: Implement input token build kernel**

The kernel must read only `State128.v[0..119]` and build `CLS + 50 piece tokens` using `fast_piece_static`, `fast_slot_projected`, `piece_positions`, and `piece_mask`.

- [ ] **Step 3: Implement LayerNorm kernels**

Use fp32 reductions over `d_model=256` with epsilon `1e-5`, matching PyTorch LayerNorm.

- [ ] **Step 4: Use CUTLASS linear helper for GEMMs**

Reuse `stream1_cutlass_linear_cuda()` for QKV, attention output projection, FFN up/down, and output head.

- [ ] **Step 5: Implement attention and SiLU kernels**

Attention is per `(row, head, query_token)` over `seq_len=51` with scale `1/sqrt(head_dim)`. SiLU is `x / (1 + exp(-x))`.

- [ ] **Step 6: Add CUDA reference test**

`tests/stream1_transformer_cuda_tests.cu` loads exported weights and `reference.json`, runs the CUDA entrypoint, and compares `score_ring` against `q_to_score_key(reference_score)` with tolerance `<= 2` score keys for fp16 T4.

---

### Task 6: Wire Transformer Backend Into Production Runner

**Files:**
- Modify: `tools/production_runner.cu`
- Modify: `cuda/dispatcher.hpp`
- Modify: `cuda/dispatcher.cu`
- Modify: `cuda/runtime_config.cpp`
- Test: `tests/dispatcher_cuda_tests.cu`

- [ ] **Step 1: Replace single MLP-only runtime fields with tagged runtime**

Add `backend`, `mlp_view`, `transformer_view`, `mlp_scratch_lanes`, and `transformer_scratch_lanes` to dispatcher network state.

- [ ] **Step 2: Dispatch explicitly in graph capture**

Inside graph capture, call exactly one of `stream1_inference_cutlass_cuda()` or `stream1_transformer_inference_cuda()` based on `network.backend`. Unknown backend is a fatal error. There is no fallback.

- [ ] **Step 3: Load weights by backend in `production_runner`**

Call `load_stream1_manifest(weight_dir)` first, then branch into MLP or transformer loader/allocation. Keep both weight/scratch lifetimes valid until after dispatcher execution.

- [ ] **Step 4: Update memory estimator**

Make `estimate_stream1_weight_bytes()` and `estimate_stream1_scratch_bytes()` switch by backend. The first transformer estimate should match the correctness-first scratch allocation exactly.

- [ ] **Step 5: Verify MLP runner still works**

Run existing small MLP smoke and confirm it prints `stream1_backend=mlp` and reaches the same early depth behavior as before.

---

### Task 7: Docker/Kaggle 2xT4 Small-Beam Smoke

**Files:**
- Create: `kaggle_t4_transformer_smoke/kernel-metadata.json`
- Create: `kaggle_t4_transformer_smoke/t4-transformer-beam-smoke.ipynb`
- Create: `test_results/stream1_transformer_kaggle_smoke_YYYY-MM-DD.md`

- [ ] **Step 1: Build Docker correctness locally**

Run:

```bash
docker run --rm --gpus all -v D:/100XH100:/workspace -w /workspace gpu-dev-cutlass-nsight bash -lc \
  'cmake -S . -B build-transformer-smoke -DBEAM_CUDA_ARCHITECTURES=75 -DCUTLASS_DIR=${CUTLASS_DIR:-/opt/cutlass} &&
   cmake --build build-transformer-smoke --target stream1_transformer_cuda_tests -j &&
   ./build-transformer-smoke/stream1_transformer_cuda_tests'
```

Expected: `stream1_transformer_cuda_tests_passed=1`.

- [ ] **Step 2: Create Kaggle package**

Use a private kernel `trydotatwo/cayley-beam-transformer-2xt4-smoke` with competition source `cayley-py-megaminx` and model source `vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1`.

- [ ] **Step 3: Notebook build/run parameters**

Use first smoke parameters:

```bash
BEAM_GLOBAL_BEAM_WIDTH=1048576
BEAM_DEPTH_LIMIT=3
BEAM_B_MICRO=512
BEAM_STREAM1_CONCURRENCY=1
BEAM_STREAM3_RING_SLOTS=2
TORCHRUN_NPROC_PER_NODE=2
BEAM_WEIGHT_DIR=/kaggle/working/stream1_transformer_weights_fp16
```

- [ ] **Step 4: Push and monitor Kaggle**

Run:

```bash
kaggle kernels push -p kaggle_t4_transformer_smoke
kaggle kernels status trydotatwo/cayley-beam-transformer-2xt4-smoke
```

Expected: kernel reaches `COMPLETE` or gives a concrete CUDA/runtime error with logs.

- [ ] **Step 5: Record verification**

Write a test result note with kernel version, status, backend, beam, depth, `B_MICRO`, `nproc_per_node`, and key log lines.

---

## Self-Review

**Spec coverage:** The plan covers a separate transformer Stream1 backend, automatic manifest-based dimensions, preservation of the existing MLP path, standalone correctness before dispatcher wiring, and a final 2xT4 small-beam Kaggle smoke.

**Placeholder scan:** No implementation step is delegated to an undefined fallback. The only deliberately non-final part is the transformer scratch layout, explicitly marked as correctness-first and bounded by small `BEAM_B_MICRO`.

**Type consistency:** The backend enum, `Stream1ModelConfig` transformer fields, `Stream1TransformerView`, and `Stream1TransformerScratchView` are named consistently across loader, CUDA entrypoint, dispatcher, runner, and tests.
