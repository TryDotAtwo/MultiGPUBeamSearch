# Stream1 Dispatcher Ring-Slot Hook

Date: 2026-07-05

Scope:

- Add a Torch-free dispatcher hook for explicit non-graph Stream1 ring-slot launchers.
- Preserve the existing default MLP/native CUDA Graph launch path when no hook is supplied.
- Do not add fallback behavior and do not link LibTorch into default `beam_cuda` builds.

Changed surface:

```text
cuda/dispatcher.hpp
cuda/dispatcher.cu
tests/dispatcher_cuda_tests.cu
```

Contract:

```text
DispatcherRingSlotLauncher = optional host callback at ring-slot scheduling boundary.
Default path = cudaGraphLaunch(graphs.ring_slot_execs[job], stream1_lane).
Custom path = caller supplies launch(context,user) and owns the external Stream1 work.
```

The hook context contains the stable scheduling data needed by a future explicit `libtorch:eager` path:

```text
job, ring, ring_slot, lane, b_micro, candidate_offset, parent_base, count, stream1_lane, stream2_lane
```

Verification:

```text
docker run --rm --gpus all -v "D:/100XH100/.worktrees/stream1-piece-transformer:/work" -w /work gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake -S . -B /tmp/build-dispatcher-hook -DCMAKE_BUILD_TYPE=Release -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75 >/tmp/cmake-dispatcher-hook.log && cmake --build /tmp/build-dispatcher-hook --target dispatcher_cuda_tests contract_tests -j2 && /tmp/build-dispatcher-hook/dispatcher_cuda_tests && /tmp/build-dispatcher-hook/contract_tests"
```

Result:

```text
dispatcher_cuda_tests=pass
contract_tests=pass
```

Runner build check:

```text
docker run --rm --gpus all -v "D:/100XH100/.worktrees/stream1-piece-transformer:/work" -w /work gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake -S . -B /tmp/build-production-hook -DCMAKE_BUILD_TYPE=Release -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75 >/tmp/cmake-production-hook.log && cmake --build /tmp/build-production-hook --target production_runner -j2"
```

Result:

```text
[100%] Built target production_runner
```

Notes:

- The first build attempt caught two compile-time issues in the draft hook (`candidate_offset` scope and unavailable dispatcher events). The final hook computes `candidate_offset` in the scheduler loop and leaves synchronization events to the future hook user-state.
- The current implementation does not yet call LibTorch from `production_runner`; it creates the explicit scheduler seam needed for that next step.