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

Dry-run a C++ LibTorch graph run:

```bash
python tools/stream1_transformer_backends.py \
  --backend libtorch \
  --mode cuda_graph \
  --build-dir build-libtorch-stream1 \
  --weight-dir /path/to/stream1_transformer_weights_fp16 \
  --batches 192,384 \
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
## Parity Gate

Use `tools/stream1_transformer_parity.py` to compare selected backends on the same synthetic state batch. This gate is explicit backend comparison, not runtime fallback behavior.

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
  --backends pytorch,libtorch,native_cutlass \
  --batch 256 \
  --tolerance 3072
```

The native CUTLASS parity path sets `BEAM_STREAM1_SYNTHETIC_STATES=1` so it uses the same deterministic arange-pattern state batch as the Torch/LibTorch benchmark paths.
