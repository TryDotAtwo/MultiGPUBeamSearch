# Stream1 Piece Transformer Three-Backend Registry

Date: 2026-07-04

Scope:

- Keep `pytorch`, `libtorch`, and `native_cutlass` as explicit selectable Stream1 `piece_transformer` execution backends.
- Do not add fallback behavior.
- Do not modify the MLP Stream1 path or production dispatcher behavior.

Changes:

- Added `tools/stream1_transformer_backends.py` as the canonical backend registry and launcher.
- Added `docs/stream1_transformer_backends.md` with backend policy and run examples.
- Updated `docs/transformer_inference_architecture.md` to replace the single-direction LibTorch note with a three-backend policy.
- Added `tests/test_stream1_transformer_backends.py` to lock backend names, modes, command generation, and fail-closed mode validation.

Verification:

```text
python tests\test_stream1_transformer_backends.py
.....
Ran 5 tests in 0.000s
OK
```

```text
$env:PYTHONPATH='.'; python tests\test_stream1_transformer_exporter.py
......
Ran 6 tests in 0.095s
OK
```

Dry-run checks:

```text
python tools\stream1_transformer_backends.py --list-backends
backend  default_mode  modes             owner              description
pytorch  eager         eager             Python/Torch       Torch implementation over exported piece_transformer weights.
libtorch eager         eager,cuda_graph  C++/LibTorch       C++ LibTorch implementation using at::linear and SDPA.
native_cutlass graph   eager,graph       C++/CUDA/CUTLASS   Native CUDA/CUTLASS Stream1 transformer path via stream_benchmark.
```

Dry-run command generation passed for:

- `--backend pytorch --weight-dir weights_fp16 --dry-run --json`
- `--backend libtorch --mode cuda_graph --build-dir build-smoke --weight-dir weights_fp16 --dry-run --json`
- `--backend native_cutlass --mode graph --build-dir build-smoke --weight-dir weights_fp16 --b-micro 256 --concurrency 2 --dry-run --json`

Result:

- The three backend families are now available behind stable names.
- Backend selection is explicit; no backend silently routes to another backend.
