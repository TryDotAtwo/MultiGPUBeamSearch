# Stream1 LibTorch Production Runner Hook

Date: 2026-07-05

Scope:

- Add an explicit opt-in `production_runner_libtorch_stream1` target behind `BEAM_ENABLE_LIBTORCH_STREAM1=ON`.
- Add `BEAM_STREAM1_EXECUTOR=libtorch_eager` runtime selection.
- Keep default `production_runner` and `beam_cuda` Torch-free.
- Do not add fallback or distillation behavior.

Implementation:

- `cuda/dispatcher.*` now lets callers skip native ring-slot graph templates when a custom `DispatcherRingSlotLauncher` is installed.
- `tools/stream1_libtorch_ring_slot_launcher.*` owns the LibTorch `piece_transformer` model, wraps dispatcher ring-slot state as CUDA tensors, writes int32 score keys into the existing score ring, and launches existing Stream2 hashing on the paired Stream2 lane.
- `tools/production_runner.cu` only loads native Stream1 device weights/scratch when the native executor is used. The LibTorch executor uses manifest-only config plus the LibTorch model loader.
- The LibTorch executor fails closed for non-`piece_transformer`, `uniform` mode, or `output_dim != MOVE_COUNT`.

Verification:

```text
git diff --check
```

Result: pass, with only existing CRLF normalization warnings for CMake/production runner files.

Default no-LibTorch Docker build in `gpu-dev-cutlass-nsight:cuda128-sm120`:

```text
cmake -S . -B /tmp/build-default-runner -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75
cmake --build /tmp/build-default-runner --target production_runner dispatcher_cuda_tests contract_tests -j2
```

Result: `production_runner`, `dispatcher_cuda_tests`, and `contract_tests` built successfully. Running `dispatcher_cuda_tests` in this local Docker failed at `cudaSetDevice(0)` because the container has no NVIDIA driver; this is an environment limitation, not a compile regression.

CPU contract check:

```text
contract_tests=pass
```

Opt-in LibTorch Docker build in `cmz-native-dev:2026-05-26`:

```text
cmake -S . -B /tmp/build-libtorch-runner \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake \
  -DBEAM_ENABLE_LIBTORCH_STREAM1=ON \
  -DCUTLASS_DIR=/opt/cutlass \
  -DBEAM_CUDA_ARCHITECTURES=75
cmake --build /tmp/build-libtorch-runner --target production_runner_libtorch_stream1 stream1_transformer_libtorch_benchmark -j2
```

Result:

```text
[100%] Built target production_runner_libtorch_stream1
[100%] Built target stream1_transformer_libtorch_benchmark
```

Compatibility note:

This is not a generic arbitrary-transformer runtime. It supports the repository's exported `piece_transformer` manifest/weight contract. Multi-output dimensions such as 18 or 24 are supported only when they match compile-time `MOVE_COUNT` and the matching generators/runtime config. Single-output transformer semantics are separate row-mode work and currently fail closed.