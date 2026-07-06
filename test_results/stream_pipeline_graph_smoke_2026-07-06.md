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

## 2026-07-06 Kaggle 2xT4 Smoke v2

Kaggle kernel `trydotatwo/cayley-beam-pipeline-smoke-2xt4` version 2 completed successfully on commit `1cd5355`. It built `stream_pipeline_benchmark` and `contract_tests`, `contract_tests=pass`, and all pipeline smoke rows returned `status=OK`.

Downloaded outputs: `test_results/kaggle_pipeline_smoke_v2_2026-07-06/`.

```text
stream12  window=16  b_micro=512 concurrency=2 ring_slots=2 stream3_batch=24576 candidates_per_sec=610063 depth_like_ms=161.137 stream3_jobs=0
stream12  window=32  b_micro=512 concurrency=2 ring_slots=2 stream3_batch=24576 candidates_per_sec=631061 depth_like_ms=155.776 stream3_jobs=0
stream123 window=16  b_micro=512 concurrency=2 ring_slots=2 stream3_batch=24576 candidates_per_sec=661734 depth_like_ms=148.555 stream3_jobs=4
stream123 window=32  b_micro=512 concurrency=2 ring_slots=2 stream3_batch=24576 candidates_per_sec=624255 depth_like_ms=157.474 stream3_jobs=4
```

Result: local runtime and Kaggle 2xT4 runtime are clean. Next step is the MEPhI 8xA100 `RUN_PIPELINE_SMOKE=1` sweep on the same branch.

## 2026-07-06 Cluster Smoke Fix 2

MEPhI cluster job 31745 used the updated benchmark but failed all rows before measurement with `FAIL_134`:

```text
pipeline_smoke_start mode=stream12 window=16 b_micro=512 concurrency=2
what(): CUDA error: cudaMemset(memory.streams.threshold_request_local, 0, sizeof(std::uint32_t))
file=.../tools/stream_pipeline_benchmark.cu line=87 message=invalid argument
```

The previous fix attached benchmark-owned `current_threshold`, `threshold_initialized`, and `current_threshold_active_index`, but the reset path also clears `threshold_request_local` and `threshold_request_global`. Those request pointers must follow the same benchmark-owned-buffer rule because the pipeline smoke tool is not a production full-depth allocation.

Fix: extend `BenchmarkThresholdBuffers` with `request_local` and `request_global`, allocate them with `cudaMalloc`, attach them to `StaticDeviceMemory`, and free them with the other benchmark threshold buffers. Production runner/static layout defaults remain unchanged.

Local Docker verification after the request-buffer fix:

```text
contract_tests=pass
stream_pipeline_benchmark mode=stream12 window=16 b_micro=64 concurrency=1 ring_slots=1 stream3_batch=1536 graph_window_jobs=2 physical_jobs=2 frontier_size=128 ring_slot_jobs=2 stream3_jobs=0 stream4_jobs=0 candidates=3072 depth_like_ms=3.47383 candidates_per_sec=884325 shard_capacity=4096 allocation_bytes=21823744 status=OK
stream_pipeline_benchmark mode=stream123 window=16 b_micro=64 concurrency=1 ring_slots=1 stream3_batch=1536 graph_window_jobs=2 physical_jobs=2 frontier_size=128 ring_slot_jobs=2 stream3_jobs=2 stream4_jobs=0 candidates=3072 depth_like_ms=10.9191 candidates_per_sec=281341 shard_capacity=4096 allocation_bytes=21823744 status=OK
```

Cluster-like local RTX smoke also passed with the A100 smoke layout (`B_MICRO=512`, concurrency `2`, ring slots `8`, rings `32`, shard count `32`, stream3 batch `98304`):

```text
stream_pipeline_benchmark mode=stream12 window=16 b_micro=512 concurrency=2 ring_slots=8 stream3_batch=98304 graph_window_jobs=32 physical_jobs=256 frontier_size=131072 ring_slot_jobs=256 stream3_jobs=0 stream4_jobs=0 candidates=3145728 depth_like_ms=3712.62 candidates_per_sec=847307 shard_capacity=1048576 allocation_bytes=3281187584 status=OK
stream_pipeline_benchmark mode=stream123 window=16 b_micro=512 concurrency=2 ring_slots=8 stream3_batch=98304 graph_window_jobs=32 physical_jobs=256 frontier_size=131072 ring_slot_jobs=256 stream3_jobs=32 stream4_jobs=0 candidates=3145728 depth_like_ms=4129.25 candidates_per_sec=761815 shard_capacity=1048576 allocation_bytes=3281187584 status=OK
```

Kaggle 2xT4 v3 should run the same cluster-like smoke dimensions with windows `16/32/64`.

## 2026-07-06 Kaggle 2xT4 Smoke v3

Kaggle kernel `trydotatwo/cayley-beam-pipeline-smoke-2xt4` version 3 completed successfully on commit `62c14bb`. It built `stream_pipeline_benchmark` and `contract_tests`, `contract_tests=pass`, and all cluster-like pipeline smoke rows returned `status=OK`.

Downloaded outputs: `test_results/kaggle_pipeline_smoke_v3_2026-07-06/`.

```text
stream12  window=16  b_micro=512 concurrency=2 ring_slots=8 stream3_batch=98304 candidates_per_sec=563956 depth_like_ms=5577.97 stream3_jobs=0 graph_window_jobs=32 physical_jobs=256 allocation_bytes=3281187584
stream12  window=32  b_micro=512 concurrency=2 ring_slots=8 stream3_batch=98304 candidates_per_sec=547178 depth_like_ms=5749    stream3_jobs=0 graph_window_jobs=64 physical_jobs=256 allocation_bytes=3281187584
stream12  window=64  b_micro=512 concurrency=2 ring_slots=8 stream3_batch=98304 candidates_per_sec=533020 depth_like_ms=5901.71 stream3_jobs=0 graph_window_jobs=128 physical_jobs=256 allocation_bytes=3281187584
stream123 window=16  b_micro=512 concurrency=2 ring_slots=8 stream3_batch=98304 candidates_per_sec=523753 depth_like_ms=6006.12 stream3_jobs=32 graph_window_jobs=32 physical_jobs=256 allocation_bytes=3281187584
stream123 window=32  b_micro=512 concurrency=2 ring_slots=8 stream3_batch=98304 candidates_per_sec=504416 depth_like_ms=6236.37 stream3_jobs=32 graph_window_jobs=64 physical_jobs=256 allocation_bytes=3281187584
stream123 window=64  b_micro=512 concurrency=2 ring_slots=8 stream3_batch=98304 candidates_per_sec=493849 depth_like_ms=6369.81 stream3_jobs=32 graph_window_jobs=128 physical_jobs=256 allocation_bytes=3281187584
```

Result: local Docker and Kaggle 2xT4 are clean after the `threshold_request_local/global` fix. The next validation step is to rerun the MEPhI 8xA100 `RUN_PIPELINE_SMOKE=1` sweep on commit `62c14bb` or newer.
