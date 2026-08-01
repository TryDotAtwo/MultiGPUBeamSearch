# Stream1 Transformer Macro/Micro Ring-Slot Fix 2026-07-06

## Problem

The previous MEPhI Megaminx transformer tuning propagated the isolated Stream1 optimum (`B_MICRO=512`) into the dispatcher runtime. That changed the external ring-slot granularity versus the MLP path and inflated ring-slot/Stream3 scheduling work in full beam-search runs.

## Fix

The dispatcher contract is restored to MLP-style external ring slots:

- `BEAM_B_MICRO` remains the external parent rows per ring-slot consumed by Stream2/3.
- `BEAM_STREAM1_TRANSFORMER_MICRO` is a new internal native PieceTransformer Stream1 slicing size.
- Native PieceTransformer Stream1 loops over `[0, B_MICRO)` in internal chunks and writes each chunk into the correct offset of the existing score ring.
- Stream2 still runs once per external ring-slot over the full `B_MICRO` score range.
- Transformer scratch allocation is sized by `BEAM_STREAM1_TRANSFORMER_MICRO`, not by the external `B_MICRO`.
- Pipeline smoke defaults now use target `B_MICRO`/concurrency instead of the isolated Stream1 micro size.

## Local Docker Verification

Image: `gpu-dev-cutlass-nsight:2026-05-24`.

Build:

```text
cmake --build build-transformer-macro-micro-check --target production_runner stream_pipeline_benchmark contract_tests stream1_transformer_cuda_tests -j2
```

Result:

```text
[100%] Built target production_runner
[100%] Built target stream_pipeline_benchmark
[100%] Built target contract_tests
[100%] Built target stream1_transformer_cuda_tests
```

Direct checks:

```text
contract_tests=pass
stream1_transformer_cuda_tests=skip missing_reference_fixture
bash_syntax_ok=1
```

Full target build plus CTest:

```text
100% tests passed, 0 tests failed out of 13
```

Macro/micro pipeline smoke on real committed Megaminx transformer weights:

```text
stream_pipeline_benchmark mode=stream123 window=16 b_micro=1024 concurrency=2 ring_slots=8 stream3_batch=196608 graph_window_jobs=8 physical_jobs=8 frontier_size=8192 ring_slot_jobs=8 stream3_jobs=1 stream4_jobs=0 candidates=196608 depth_like_ms=277.974 candidates_per_sec=707290 shard_capacity=1048576 allocation_bytes=3234788864 status=OK
stream_pipeline_benchmark mode=stream12 window=16 b_micro=8192 concurrency=2 ring_slots=8 stream3_batch=1572864 graph_window_jobs=8 physical_jobs=8 frontier_size=65536 ring_slot_jobs=8 stream3_jobs=0 stream4_jobs=0 candidates=1572864 depth_like_ms=2405.99 candidates_per_sec=653727 shard_capacity=1572864 allocation_bytes=5091152640 status=OK
stream_pipeline_benchmark mode=stream123 window=16 b_micro=8192 concurrency=2 ring_slots=8 stream3_batch=1572864 graph_window_jobs=8 physical_jobs=8 frontier_size=65536 ring_slot_jobs=8 stream3_jobs=1 stream4_jobs=0 candidates=1572864 depth_like_ms=2482.8 candidates_per_sec=633504 shard_capacity=1572864 allocation_bytes=5091152640 status=OK
```

## Notes

The local throughput is not the A100 target number; the purpose of this smoke was to prove the corrected external ring-slot shape and Stream3 compatibility before rerunning MEPhI jobs.
## Pipeline Smoke Launcher Follow-up

Cluster job `31756` still failed because `BEAM_STREAM3_BATCH_CANDIDATES` could be inherited from the submit shell and override `BEAM_STREAM3_RING_SLOTS`, deriving an effective `RING_SLOT_COUNT` smaller than `BEAM_STREAM1_CONCURRENCY`.

`run_pipeline_smoke_config` now computes and passes an explicit smoke-local Stream3 batch:

```text
stream3_batch = b_micro * 24 * BEAM_STREAM3_RING_SLOTS
```

For the intended macro/micro test this is:

```text
8192 * 24 * 8 = 1572864
```

Verification:

```text
stream_pipeline_benchmark mode=stream123 window=64 b_micro=8192 concurrency=8 ring_slots=8 stream3_batch=1572864 graph_window_jobs=8 physical_jobs=8 frontier_size=65536 ring_slot_jobs=8 stream3_jobs=1 stream4_jobs=0 candidates=1572864 depth_like_ms=2335.81 candidates_per_sec=673369 shard_capacity=1572864 allocation_bytes=5091152640 status=OK
```

## Full Target Launcher Follow-up

Cluster job `31758` failed before launching the solver:

```text
invalid_stream1_concurrency=8 stream3_effective_ring_slots=4
```

Cause: the full target path could also inherit a stale `BEAM_STREAM3_BATCH_CANDIDATES` from the submit shell. That stale value overrode `BEAM_STREAM3_RING_SLOTS=8` and made the helper derive only four effective Stream3 ring slots.

Fix: `run_full_config` now sets a run-local `BEAM_STREAM3_BATCH_CANDIDATES` before deriving/validating manual config:

```text
BEAM_STREAM3_BATCH_CANDIDATES = b_micro * 24 * ring_slots
```

For the current 700M Megaminx transformer target:

```text
8192 * 24 * 8 = 1572864
```

A custom full-run value can still be supplied through `FULL_STREAM3_BATCH_CANDIDATES`; stale generic shell env no longer changes target/smoke full-run slot geometry.

Verification:

```text
bash_syntax_ok=1
```

## Runtime Budget Estimate Follow-up 2026-07-07

Cluster job `31759` reached production runner startup but failed the manual GPU budget guard:

```text
manual runtime config exceeds GPU budget: required=58475547300 budget=41100640256 free_before=41163751168 headroom=536870912
```

The `s31572864` token in the log path is the tag prefix `s3` plus `1572864`; Stream3 batch was already correct.

Cause: `estimate_non_static_device_bytes` still estimated PieceTransformer Stream1 scratch using external `B_MICRO=8192`, while the actual production runner allocation had already been changed to use internal `BEAM_STREAM1_TRANSFORMER_MICRO=512`. The guard was therefore rejecting a configuration based on scratch memory that would not be allocated.

Fix: Runtime config budget estimation now uses `BEAM_STREAM1_TRANSFORMER_MICRO` for PieceTransformer scratch byte estimates, with the same `[1, B_MICRO]` validation as production allocation. MLP estimates still use external `B_MICRO`.

Verification:

```text
100% tests passed, 0 tests failed out of 13
stream_pipeline_benchmark mode=stream123 window=64 b_micro=8192 concurrency=8 ring_slots=8 stream3_batch=1572864 graph_window_jobs=8 physical_jobs=8 frontier_size=65536 ring_slot_jobs=8 stream3_jobs=1 stream4_jobs=0 candidates=1572864 depth_like_ms=2517.36 candidates_per_sec=624808 shard_capacity=1572864 allocation_bytes=5091152640 status=OK
```
