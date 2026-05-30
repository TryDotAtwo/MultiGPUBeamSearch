# Stream3/Stream4 batch guard 2026-05-30

task_id=stream3_stream4_batch_guard
environment=Docker service beam-tests, CUDA 12.4.1 image gpu-dev-cutlass-nsight:2026-05-24

## Change

- Removed the invalid runtime guard requiring `STREAM4_BATCH_CANDIDATES >= STREAM3_BATCH_CANDIDATES`.
- Added the actual physical-shard safety guard: `STREAM3_BATCH_CANDIDATES <= SHARD_CAPACITY_CANDIDATES`.
- Kept existing guards requiring `STREAM4_BATCH_CANDIDATES <= SHARD_CAPACITY_CANDIDATES` and `STREAM4_TRIGGER_CANDIDATES <= SHARD_CAPACITY_CANDIDATES`.
- Updated Kaggle preflight to use the same Stream3-vs-physical-shard guard.
- Updated Kaggle preflight and derived config display to convert `B_MICRO` row budget into parent batch when local model weights infer `output_dim=1`.
- Updated Kaggle local model handling so a directory path under `/kaggle/input/...` resolves to the first `.pth` file recursively.

## Verification

```text
command=python read-only notebook code compile check for both Kaggle notebooks
result=pass
evidence=notebook_compile=pass for cayley-beam-gpu-runner.ipynb and beam_kernel.ipynb
```

```text
command=rg -n "must be at least STREAM3_BATCH|must be >= STREAM3_BATCH|STREAM3_BATCH_CANDIDATES=.*must be <= SHARD_CAPACITY|STREAM3_BATCH_CANDIDATES must not exceed" cuda kaggle
result=pass
evidence=old_guard_absent; new_guard_present_in_runtime_and_preflight
```

```text
command=docker compose run --rm beam-tests bash -lc "cmake --build build-docker --target production_runner -j 2 && BEAM_RUNTIME_CONFIG_MODE=manual BEAM_B_MICRO=32 BEAM_STREAM1_CONCURRENCY=1 BEAM_STREAM3_RING_SLOTS=2 BEAM_SHARD_COUNT=4 BEAM_SHARD_BUFFER_COUNT=2 BEAM_STREAM4_BATCH_ALIGNMENT=1024 BEAM_SHARD_CAPACITY_CANDIDATES=2048 BEAM_STREAM4_BATCH_CANDIDATES=1024 BEAM_STREAM4_TRIGGER_CANDIDATES=1024 BEAM_STREAM4_ACTIVE_SORT_SLOTS=1 BEAM_GLOBAL_SPILL_CAPACITY=0 BEAM_GPU_HEADROOM_BYTES=268435456 build-docker/production_runner 0 1 4096"
result=pass
evidence=STREAM3_BATCH_CANDIDATES=1536; STREAM4_BATCH_CANDIDATES=1024; SHARD_CAPACITY_CANDIDATES=2048; completed_depths=1
```

```text
command=git diff --check
result=pass
```
