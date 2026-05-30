# Stream1 output_dim=1 support 2026-05-30

task_id=stream1_output_dim1_parent_aligned_batch
branch=main
environment=Docker service beam-tests, CUDA 12.4.1 image gpu-dev-cutlass-nsight:2026-05-24

## Change

- Allowed Stream1 model manifests with `output_dim=1` or `output_dim=24`.
- Preserved downstream candidate layout: `config.b_micro` remains parent count for Stream2/3/4/5.
- Interpreted `BEAM_B_MICRO` as Stream1 row budget before runtime config creation; for `output_dim=1`, `config.b_micro=floor(BEAM_B_MICRO/24)`.
- Added Stream1 child-feature path: row `i` maps to `parent=i/24`, `move=i%24`, reads parent state, applies generator in the input-folding kernel, and writes score order `parent0_move0..23`.
- Avoided writing child `State128` records to VRAM.
- Kept hidden/residual MLP layers on the existing CUTLASS path. `output_dim=1` uses a direct final-head quantization kernel because the existing TensorOp GEMM path with `N=1` caused a misaligned-address failure in the focused test.

## Verification

```text
command=docker compose run --rm beam-tests bash -lc "cmake --build build-docker --target stream1_cuda_tests production_runner stream_benchmark -j 2"
result=pass
```

```text
command=docker compose run --rm beam-tests bash -lc "cmake --build build-docker --target stream1_cuda_tests production_runner stream_benchmark -j 2 && build-docker/stream1_cuda_tests"
result=pass
evidence=stream1_cuda_tests=pass
```

```text
command=docker compose run --rm beam-tests bash -lc "BEAM_RUNTIME_CONFIG_MODE=manual BEAM_B_MICRO=32 BEAM_STREAM1_CONCURRENCY=1 BEAM_STREAM3_RING_SLOTS=1 BEAM_SHARD_COUNT=4 BEAM_SHARD_BUFFER_COUNT=2 BEAM_STREAM4_BATCH_ALIGNMENT=1024 BEAM_SHARD_CAPACITY_CANDIDATES=2048 BEAM_STREAM4_BATCH_CANDIDATES=1024 BEAM_STREAM4_TRIGGER_CANDIDATES=1024 BEAM_STREAM4_ACTIVE_SORT_SLOTS=1 BEAM_GLOBAL_SPILL_CAPACITY=0 BEAM_GPU_HEADROOM_BYTES=268435456 build-docker/production_runner 0 1 4096"
result=pass
evidence=completed_depths=1; solution_found=0; stream1_model_output_dim=24; STREAM1_ROWS_PER_JOB=32
```

```text
command=git diff --check
result=pass
```

## Notes

- Stream2, Stream3, Stream4, and Stream5 kernel logic was not changed.
- Dispatcher change is limited to passing the read-only generator table into the Stream1 launch.
- Remaining untested path: full production run with a real exported `output_dim=1` weight directory.
