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