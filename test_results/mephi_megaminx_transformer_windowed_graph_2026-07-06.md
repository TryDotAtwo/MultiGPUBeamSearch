# MEPhI Megaminx Transformer Windowed Graph Smoke

Date: 2026-07-06

Scope:

- Add a bounded CUDA Graph executor window for native Stream1 transformer runs.
- Keep the explicit `native_eager` no-graph path unchanged.
- Keep existing default MLP/native graph behavior unchanged unless `BEAM_RING_GRAPH_EXECS_PER_LANE` is set.

Runtime control:

```text
BEAM_RING_GRAPH_EXECS_PER_LANE=<N>
```

The runtime derives the maximum physical ring count from the configured `STREAM3_RING_SLOTS` and `BEAM_STREAM1_CONCURRENCY`. For the current cluster transformer test shape:

```text
BEAM_STREAM3_RING_SLOTS=8
BEAM_STREAM1_CONCURRENCY=2
BEAM_RING_GRAPH_EXECS_PER_LANE=32
```

Expected production log:

```text
runtime_ring_count=8
runtime_ring_slot_count=8
runtime_ring_slot_graph_execs=64
runtime_ring_slot_graph_execs_per_lane=32
```

Launcher aliases:

```text
MEGAMINX_STREAM1_BACKEND=native_windowed_graph
MEGAMINX_STREAM1_BACKEND=native_graph_window
```

Both aliases compile/use `BEAM_STREAM1_EXECUTOR=native_cuda_graph` and default `BEAM_RING_GRAPH_EXECS_PER_LANE=32`.

Cluster baseline that motivated this change:

```text
native_eager 700M p990 depth 7: depth_sec=328.366, ring_slot_jobs=80230
80230 * 512 * 24 / 328.366 = 3.002M candidates/s
old MLP 700M depth step was about 85s, roughly 11.6M candidates/s at the same frontier size
isolated transformer native graph best was 4.46M candidates/s at b_micro=512 concurrency=2
```

Verification:

```text
bash -n hpc/start_8xa100_libtorch_megaminx.sh hpc/solve_then_reflect.sh
cmake -S . -B build-windowed-graph-check -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75
cmake --build build-windowed-graph-check --target production_runner contract_tests -j2
./build-windowed-graph-check/contract_tests
```

Result:

```text
[100%] Built target production_runner
[100%] Built target contract_tests
contract_tests=pass
```