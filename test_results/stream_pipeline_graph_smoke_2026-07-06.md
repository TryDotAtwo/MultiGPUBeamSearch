# Stream Pipeline Graph Smoke 2026-07-06

## Scope

Implemented a dispatcher-backed smoke benchmark for the native Stream1 piece-transformer graph path:

- `stream12`: Stream1 native piece-transformer CUDA Graph plus Stream2 hash/goal in the real ring-slot dispatcher graph.
- `stream123`: the same ring-slot graph path plus the real Stream3 ring graph, stopping before Stream4/Stream5/history/final.
- `BEAM_RING_GRAPH_EXECS_PER_LANE` is measured as a graph-template window (`16/32/64`) and does not alter Stream3 slots or production defaults.

Production behavior remains the default `DepthDispatchStopStage::Full`.

## Changed Files

- `cuda/dispatcher.hpp`: added `DepthDispatchStopStage` enum and optional `stop_stage` argument defaulting to `Full`.
- `cuda/dispatcher.cu`: added benchmark-only `AfterStream12` and `AfterStream3` stop points, plus a fail-fast guard if Stream3 benchmark mode hits shard backpressure without Stream4.
- `tools/stream_pipeline_benchmark.cu`: new single-process smoke tool using real static memory, dispatcher graphs, native piece-transformer Stream1, Stream2, and optionally Stream3.
- `CMakeLists.txt`: new `stream_pipeline_benchmark` target.
- `hpc/bench_8xa100_megaminx_transformer.sh`: new `RUN_PIPELINE_SMOKE=1` mode and TSV summary.

## Local Verification

Docker image: `gpu-dev-cutlass-nsight:2026-05-24`.

Build command:

```bash
cmake -S . -B build-stream-pipeline-smoke -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75
cmake --build build-stream-pipeline-smoke --target production_runner stream_pipeline_benchmark contract_tests stream2_cuda_tests dispatcher_cuda_tests -j2
```

Result:

```text
Built target production_runner
Built target stream_pipeline_benchmark
Built target contract_tests
Built target stream2_cuda_tests
Built target dispatcher_cuda_tests
```

Focused rebuild after cleanup:

```bash
cmake --build build-stream-pipeline-smoke --target stream_pipeline_benchmark -j2
```

Result:

```text
Built target stream_pipeline_benchmark
```

CPU contract check:

```bash
./build-stream-pipeline-smoke/contract_tests
```

Result:

```text
contract_tests=pass
```

Shell syntax check:

```bash
bash -n hpc/bench_8xa100_megaminx_transformer.sh
```

Result: pass in Docker `bash:5.2` after normalizing one stray CR in the script.

Note: local Docker has no NVIDIA driver, so CUDA runtime execution of `stream_pipeline_benchmark`, `stream2_cuda_tests`, and `dispatcher_cuda_tests` must run on the A100 cluster.

## Cluster Smoke Commands

Copy/update the script after pulling this branch on the cluster, then run one short `stream12 window=32` smoke first:

```bash
cd /mnt/pool/6/vokirova/beam8a100/repo
git fetch origin codex/stream1-piece-transformer --depth 1
git checkout -B codex/stream1-piece-transformer origin/codex/stream1-piece-transformer
git log -1 --oneline

cd /mnt/pool/6/vokirova/beam8a100
cp repo/hpc/bench_8xa100_megaminx_transformer.sh .
sed -i 's/\r$//' bench_8xa100_megaminx_transformer.sh
bash -n bench_8xa100_megaminx_transformer.sh
chmod +x bench_8xa100_megaminx_transformer.sh

export RUN_ISOLATED_STREAM1=0
export RUN_PIPELINE_SMOKE=1
export PIPELINE_SMOKE_MODES=stream12
export PIPELINE_GRAPH_WINDOW_SWEEP=32
export PIPELINE_B_MICRO_SWEEP=512
export PIPELINE_CONCURRENCY_SWEEP=2

sbatch -p kaf12 --export=ALL bench_8xa100_megaminx_transformer.sh
```

Full requested sweep after the first smoke is clean:

```bash
export RUN_ISOLATED_STREAM1=0
export RUN_PIPELINE_SMOKE=1
export PIPELINE_SMOKE_MODES="stream12 stream123"
export PIPELINE_GRAPH_WINDOW_SWEEP="16 32 64"
export PIPELINE_B_MICRO_SWEEP=512
export PIPELINE_CONCURRENCY_SWEEP=2

sbatch -p kaf12 --export=ALL bench_8xa100_megaminx_transformer.sh
```

TSV path printed by the job:

```text
logs/tuning_<jobid>/megaminx_transformer_pipeline_smoke_<jobid>.tsv
```

Expected columns:

```text
mode,window,b_micro,concurrency,ring_slots,stream3_batch,graph_window_jobs,physical_jobs,candidates_per_sec,depth_like_ms,status,log
```

## Interpretation

- `stream12` near the isolated native Stream1 baseline (`~4.476M cand/s`) means Stream2 plus graph-window scheduling is fine.
- `stream12` far below baseline points at graph-window scheduling, job-index memcpy/event synchronization, or Stream2 graph work.
- `stream123` far below `stream12` points at Stream3.
- If `stream123` is healthy while full solve is slow, continue profiling Stream4/5/history/finalization.

## 2026-07-06 Cluster Smoke Fix

Cluster job 31743 reached the new pipeline-smoke path, but the first `stream12 window=32 b_micro=512 concurrency=2` run aborted before measurement:

```text
pipeline_smoke_start mode=stream12 window=32 b_micro=512 concurrency=2
what(): CUDA error: cudaMemset(memory.streams.current_threshold_active_index, 0, sizeof(std::uint32_t))
file=.../tools/stream_pipeline_benchmark.cu line=86 message=invalid argument
```

The CMake/Torch warnings in the same log are build noise; the real failure was inside the new benchmark reset path. Existing stream microbenchmarks allocate the threshold triple through `BenchmarkThresholdBuffers` and attach it to `StaticDeviceMemory` before touching Stream3/4 threshold kernels. The new pipeline smoke tool used the static layout threshold pointers directly, which is not the established benchmark pattern.

Fix: `stream_pipeline_benchmark` now allocates and attaches benchmark threshold buffers before reset/graph instantiation, then frees them before `free_static_device_memory`.

Local Docker verification after the fix:

```bash
cmake -S . -B build-stream-pipeline-smoke-fix -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75
cmake --build build-stream-pipeline-smoke-fix --target stream_pipeline_benchmark contract_tests -j2
./build-stream-pipeline-smoke-fix/contract_tests
```

Result:

```text
[100%] Built target stream_pipeline_benchmark
[100%] Built target contract_tests
contract_tests=pass
```

A100 runtime smoke still needs to be re-run on the cluster after local/Kaggle checks.

## 2026-07-06 Local RTX Runtime Smoke

Local Docker GPU access was rechecked with `docker run --gpus all ... nvidia-smi` and found one RTX 3070 Laptop GPU with CUDA 12.8 driver support. The pipeline smoke tool was rebuilt for `sm_86` and run with a small local-safe config:

```bash
cmake -S . -B build-stream-pipeline-local86 -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=86
cmake --build build-stream-pipeline-local86 --target stream_pipeline_benchmark contract_tests -j2
./build-stream-pipeline-local86/contract_tests
```

Runtime smoke:

```text
stream_pipeline_benchmark mode=stream12 window=16 b_micro=64 concurrency=1 ring_slots=1 stream3_batch=1536 graph_window_jobs=2 physical_jobs=2 frontier_size=128 ring_slot_jobs=2 stream3_jobs=0 stream4_jobs=0 candidates=3072 depth_like_ms=3.5295 candidates_per_sec=870379 shard_capacity=4096 allocation_bytes=32162304 status=OK
stream_pipeline_benchmark mode=stream123 window=16 b_micro=64 concurrency=1 ring_slots=1 stream3_batch=1536 graph_window_jobs=2 physical_jobs=2 frontier_size=128 ring_slot_jobs=2 stream3_jobs=2 stream4_jobs=0 candidates=3072 depth_like_ms=3.90817 candidates_per_sec=786045 shard_capacity=4096 allocation_bytes=32162304 status=OK
```

Result: local runtime path is clean for `stream12` and `stream123`; next validation target is Kaggle 2xT4 with the new script package `kaggle_t4_pipeline_smoke`.

## 2026-07-06 Kaggle 2xT4 Smoke v1

Kaggle kernel `trydotatwo/cayley-beam-pipeline-smoke-2xt4` version 1 built `stream_pipeline_benchmark` and passed `contract_tests`, but all four runtime rows failed before graph execution:

```text
what(): STREAM1_CONCURRENCY must be in [1, RING_SLOT_COUNT]
```

Root cause: the package used `CONCURRENCY=2` with `RING_SLOTS=1`. This is a config error caught by the runtime guard, not a CUDA graph/runtime failure. Downloaded outputs are under `test_results/kaggle_pipeline_smoke_v1_2026-07-06/`.

Fix for v2: set `RING_SLOTS=2`, add a package preflight guard for `CONCURRENCY <= RING_SLOTS`, and run `contract_tests` from the cloned repo under `/tmp` so Kaggle outputs do not include contract-test fixture files.
