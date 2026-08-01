# Stream1 Transformer LibTorch Backend Split Smoke

Date: 2026-07-04

Scope:

- Split the opt-in C++ LibTorch Stream1 piece-transformer benchmark into a reusable backend scaffold plus a CLI/timing wrapper.
- Keep default builds Torch-free: `BEAM_ENABLE_LIBTORCH_STREAM1=OFF` remains the default.
- Keep MLP/native CUDA paths separate and do not add fallback or distillation behavior.
- Keep Stream1/Kaggle weights out of Git/worktree; tests must not depend on root `stream1_weights`.

Changed shape:

```text
tools/stream1_transformer_libtorch_backend.hpp   reusable LibTorch piece-transformer loader/forward helpers
tools/stream1_transformer_libtorch_backend.cpp   backend translation unit
tools/stream1_transformer_libtorch_benchmark.cpp CLI benchmark wrapper only
```

Build checks:

```text
cmake -S . -B build-no-libtorch-refactor2 \
  -DCUTLASS_DIR=/opt/cutlass \
  -DBEAM_CUDA_ARCHITECTURES=75
cmake --build build-no-libtorch-refactor2 --target contract_tests -j2
./build-no-libtorch-refactor2/contract_tests
```

Result:

```text
BEAM_ENABLE_LIBTORCH_STREAM1:BOOL=OFF
contract_tests=pass
```

The clean no-LibTorch contract test now writes a tiny synthetic MLP fixture under `test_results/stream1_mlp_weight_loading_ok` instead of requiring checked-in root weights.

Opt-in LibTorch target build:

```text
cmake -S . -B build-libtorch-refactor2 \
  -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake \
  -DBEAM_ENABLE_LIBTORCH_STREAM1=ON \
  -DCUTLASS_DIR=/opt/cutlass \
  -DBEAM_CUDA_ARCHITECTURES=75
cmake --build build-libtorch-refactor2 --target stream1_transformer_libtorch_benchmark -j2
```

Result:

```text
BEAM_ENABLE_LIBTORCH_STREAM1:BOOL=ON
Torch_DIR:PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch
[100%] Built target stream1_transformer_libtorch_benchmark
```

Notes:

- Runtime CPU/GPU forward smoke was not rerun in this split check because the Stream1/Kaggle weight directories were intentionally deleted from the worktree. The previous fp16 CPU forward smoke remains in `test_results/stream1_transformer_libtorch_backend_smoke_2026-07-04.md`.
- This is still an explicit benchmark/backend scaffold. Production dispatcher integration remains separate and must make an explicit graph-capture/non-graph execution decision.