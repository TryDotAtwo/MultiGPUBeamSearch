# Ring Graph Debug Sync Check

Date: 2026-07-08

Context:

- Cluster run `31774` used Megaminx transformer native CUDA graph with `BEAM_RING_GRAPH_EXECS_PER_LANE=8`.
- Runtime confirmed `runtime_ring_slot_graph_windowed=1`, `window_rings=8`, `window_jobs=64`.
- The run still aborted on depth 7 with only torchrun `SIGABRT` in stdout/rank logs; no CUDA root cause was printed.

Change:

- Added opt-in `BEAM_DEBUG_RING_GRAPH_SYNC=1` diagnostics in the windowed ring-slot graph path.
- When enabled, the dispatcher synchronizes the lane stream immediately after each windowed graph launch and prints rank/ring/slot/job/template_job/parent_base/count before throwing the CUDA error.
- Default behavior is unchanged when the env var is unset.

Verification:

```text
docker run --rm -v D:\100XH100\.worktrees\stream1-piece-transformer:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake -S . -B build-ring-debug-check -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=80 -DBEAM_ENABLE_LIBTORCH_STREAM1=OFF >/tmp/cmake.log && cmake --build build-ring-debug-check --target production_runner stream_pipeline_benchmark -j2"
```

Result:

```text
[100%] Built target production_runner
[100%] Built target stream_pipeline_benchmark
```