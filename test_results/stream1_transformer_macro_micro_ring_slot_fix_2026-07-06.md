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