# MEPhI Megaminx Transformer Dynamic Windowed Graph

Date: 2026-07-06

Scope:

- Replace the first windowed graph implementation that capped physical `ring_count`.
- Keep full physical ring pool, Stream3 ring slots, and static memory plan unchanged.
- Bound only the number of instantiated Stream1+Stream2 ring-slot CUDA graph templates.

Design:

```text
BEAM_RING_GRAPH_EXECS_PER_LANE=32
```

For `STREAM3_RING_SLOTS=8` and `BEAM_STREAM1_CONCURRENCY=2`, this creates:

```text
slots_per_lane = ceil(8 / 2) = 4
window_rings = 32 / 4 = 8
window_jobs = 8 * 8 = 64
```

The physical `runtime_ring_count` remains the value derived from the shard size. Only `runtime_ring_slot_graph_window_jobs` is capped.

Implementation notes:

- `CudaGraphJobTemplates` now stores a bounded ring-slot graph window plus a device `ring_slot_job_index` array.
- Windowed PieceTransformer graph kernels read `job = *graph_job_index` on device.
- Stream1 input kernels read `parent_base[job]` / `count[job]`.
- Stream1 score quantize writes to `score_ring + job * b_micro * MOVE_COUNT`.
- Stream2 hash writes to `hash_ring + job * b_micro * MOVE_COUNT`.
- The scheduler copies the physical `job` into the selected graph template before `cudaGraphLaunch`.
- Template reuse is guarded by per-template completion events.
- Existing MLP/default full-graph behavior remains unchanged when `BEAM_RING_GRAPH_EXECS_PER_LANE` is unset.

Expected cluster log shape:

```text
runtime_ring_count=<full derived ring_count>
runtime_ring_slot_count=8
runtime_ring_slot_physical_jobs=<runtime_ring_count * 8>
runtime_ring_graph_execs_per_lane_requested=32
runtime_ring_slot_graph_windowed=1
runtime_ring_slot_graph_window_rings=8
runtime_ring_slot_graph_window_jobs=64
runtime_ring_slot_graph_physical_jobs=<same as physical_jobs>
```

Verification:

```text
bash -n hpc/start_8xa100_libtorch_megaminx.sh hpc/solve_then_reflect.sh
cmake -S . -B build-windowed-graph-check -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75
cmake --build build-windowed-graph-check --target production_runner contract_tests -j2
./build-windowed-graph-check/contract_tests
cmake --build build-windowed-graph-check --target stream1_transformer_cuda_tests stream2_cuda_tests dispatcher_cuda_tests -j2
```

Result:

```text
[100%] Built target production_runner
contract_tests=pass
[100%] Built target stream1_transformer_cuda_tests
[100%] Built target stream2_cuda_tests
[100%] Built target dispatcher_cuda_tests
```

GPU runtime still needs the MEPhI A100 smoke run because this local Docker environment has no NVIDIA driver attached.