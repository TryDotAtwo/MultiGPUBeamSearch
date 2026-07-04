# Stream1 LibTorch Slot-Cache Build Check

Date: 2026-07-04

Change:

- Precompute per-slot `piece_positions`, `fast_slot_projected[slot]`, and dtype-cast slot masks once at model load.
- Keep transformer architecture and LibTorch graph mode unchanged.
- MLP/default build remains Torch-free.

Verification:

```text
cmake -S /workspace -B /tmp/build-libtorch-slotcache \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake \
  -DBEAM_ENABLE_LIBTORCH_STREAM1=ON \
  -DCUTLASS_DIR=/opt/cutlass \
  -DBEAM_CUDA_ARCHITECTURES=75
cmake --build /tmp/build-libtorch-slotcache --target stream1_transformer_libtorch_benchmark -j2
```

Result: `Built target stream1_transformer_libtorch_benchmark`.

```text
cmake -S /workspace -B /tmp/build-no-libtorch-slotcache \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_DIR=/opt/cutlass \
  -DBEAM_CUDA_ARCHITECTURES=75
cmake --build /tmp/build-no-libtorch-slotcache --target contract_tests -j2
/tmp/build-no-libtorch-slotcache/contract_tests
```

Result: `contract_tests=pass`.

CPU numerical smoke was skipped because local reference transformer weights were already removed: `reference_weights_missing=1`.