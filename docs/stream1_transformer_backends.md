# Stream1 Piece Transformer Backends

Date: 2026-07-04

## Policy

The Stream1 `piece_transformer` work keeps three explicit execution backends:

| Backend | Owner | Modes | Purpose |
|---|---|---|---|
| `pytorch` | Python/Torch | `eager` | Fast Python reference and Kaggle notebook path. |
| `libtorch` | C++/LibTorch | `eager`, `cuda_graph` | Explicit C++ Torch execution backend over exported weights. |
| `native_cutlass` | C++/CUDA/CUTLASS | `eager`, `graph` | Native production-oriented CUDA backend. |

These are not fallbacks. A run must select one backend explicitly. If the selected backend is missing dependencies, the run should fail loudly instead of silently routing to another backend.

The MLP Stream1 path stays separate and unchanged.

## Canonical Launcher

Use `tools/stream1_transformer_backends.py` as the shared backend registry:

```bash
python tools/stream1_transformer_backends.py --list-backends
```

Dry-run a PyTorch benchmark over exported weights:

```bash
python tools/stream1_transformer_backends.py \
  --backend pytorch \
  --weight-dir /path/to/stream1_transformer_weights_fp16 \
  --batches 2048,4096 \
  --dry-run
```

Dry-run the current measured C++ LibTorch T4 candidate:

```bash
python tools/stream1_transformer_backends.py \
  --backend libtorch \
  --mode eager \
  --build-dir build-libtorch-stream1 \
  --weight-dir /path/to/stream1_transformer_weights_fp16 \
  --batches 384 \
  --passes 3 \
  --dry-run
```

Dry-run native CUTLASS graph mode:

```bash
python tools/stream1_transformer_backends.py \
  --backend native_cutlass \
  --mode graph \
  --build-dir build-transformer \
  --weight-dir /path/to/stream1_transformer_weights_fp16 \
  --b-micro 256 \
  --concurrency 2 \
  --dry-run
```

Remove `--dry-run` to execute the selected backend.

## Build Boundaries

`pytorch` needs Python Torch and exported fp16 weights.

`libtorch` is opt-in. Build it with:

```bash
cmake -S . -B build-libtorch-stream1 \
  -DBEAM_ENABLE_LIBTORCH_STREAM1=ON \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DCUTLASS_DIR=/opt/cutlass \
  -DBEAM_CUDA_ARCHITECTURES=75
cmake --build build-libtorch-stream1 --target stream1_transformer_libtorch_benchmark -j2
cmake --build build-libtorch-stream1 --target production_runner_libtorch_stream1 -j2
```

`native_cutlass` uses the existing `stream_benchmark` target:

```bash
cmake -S . -B build-transformer \
  -DCUTLASS_DIR=/opt/cutlass \
  -DBEAM_CUDA_ARCHITECTURES=75
cmake --build build-transformer --target stream_benchmark -j2
```

## Kaggle 2xT4

The private Kaggle package `kaggle_t4_transformer_backend_compare/` currently runs all three families in one job:

- original PyTorch `batch_process`;
- Python Torch over exported weights;
- C++ LibTorch eager/CUDA Graph;
- native CUTLASS eager/graph.

For strategy decisions, use the aggregate backend summary from that package. For production integration work, keep the three backend families independently selectable through the launcher above.
## Current T4 Candidate

Kaggle 2xT4 v11 repeated-pass stability confirmed explicit C++ LibTorch eager as the current exported-weight transformer candidate. With three passes per batch, `batch=384` was the mean-best point on both T4s: aggregate mean `1467441.7` candidates/s, about `1.032x` of the full-compare v5 original PyTorch `batch_process` reference. The best single v11 eager rows aggregated to `1497577.0` candidates/s. CUDA Graph capture works as an explicit benchmark mode, but it remained slower than eager: aggregate mean `1411365.0` candidates/s and best aggregate `1442113.0`.

Use repeated passes when tuning this backend because single rows are noisy on Kaggle T4:

```bash
python tools/stream1_transformer_backends.py \
  --backend libtorch \
  --mode eager \
  --build-dir build-libtorch-stream1 \
  --weight-dir /path/to/stream1_transformer_weights_fp16 \
  --batches 384 \
  --passes 3
```

This is a measured benchmark candidate, not a fallback route. Production integration should select `libtorch:eager` explicitly and fail loudly if LibTorch is unavailable.
## Parity Gate

Use `tools/stream1_transformer_parity.py` to compare selected backends on the same synthetic state batch. This gate is explicit backend comparison, not runtime fallback behavior. Real runs require every backend to emit `score_key_digest`, a stable FNV-1a 64-bit digest over every quantized score key in batch-major, move-major order. Cross-backend pass/fail uses `first_score_keys` tolerance by default because PyTorch, LibTorch, and native CUTLASS can choose different valid FP16 kernel orders; add `--require-exact-digest` when comparing backend-family variants that must be bit-exact.

Dry-run the parity plan without requiring weights or built binaries:

```bash
python tools/stream1_transformer_parity.py \
  --weight-dir /path/to/stream1_transformer_weights_fp16 \
  --build-dir build-transformer \
  --backends pytorch,libtorch,native_cutlass \
  --batch 256 \
  --dry-run
```

Run the parity gate after the three backend tools are available:

```bash
python tools/stream1_transformer_parity.py \
  --weight-dir /path/to/stream1_transformer_weights_fp16 \
  --build-dir build-transformer \
  --backends pytorch,libtorch:cuda_graph,native_cutlass:graph \
  --batch 256 \
  --tolerance 3072
```

The native CUTLASS parity path sets `BEAM_STREAM1_SYNTHETIC_STATES=1` so it uses the same deterministic arange-pattern state batch as the Torch/LibTorch benchmark paths.
Backend entries accept `backend:mode`; a bare backend defaults to `eager`. Examples: `libtorch:cuda_graph`, `native_cutlass:graph`, and `pytorch`.
## Production Runner Selection

The default production binary remains the normal Torch-free target:

```bash
cmake --build build-transformer --target production_runner -j2
```

To run Stream1 through C++ LibTorch, build the explicit target and set the executor env:

```bash
cmake --build build-libtorch-stream1 --target production_runner_libtorch_stream1 -j2
BEAM_STREAM1_EXECUTOR=libtorch_eager \
BEAM_WEIGHT_DIR=/path/to/stream1_transformer_weights_fp16 \
./build-libtorch-stream1/production_runner_libtorch_stream1 <puzzle_id> <depth> <beam> [world_size] [local_rank]
```

`libtorch_eager` currently supports the exported `piece_transformer` contract only when the model's `output_dim` equals compile-time `MOVE_COUNT`. A 24-output model works in a 24-move build; an 18-output model requires an 18-move build and matching generators/runtime config. Single-output transformer row-mode is intentionally not wired yet and fails closed instead of being guessed.
