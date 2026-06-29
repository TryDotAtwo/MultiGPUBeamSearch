# Stream1 Transformer Runtime Wiring 2026-06-29

## Scope
- Wired Stream1 runtime dispatcher and production runner to select explicit Stream1 backends: MLP or piece_transformer.
- Kept `BEAM_STREAM1_MODE=uniform` as a model-inference skip path for both backends.
- Kept `stream_benchmark` MLP-only with an explicit fail-closed message for piece_transformer.
- Did not change State128, Stream3, Stream4, or exported weight format contracts.

## Red Test
- Added a dispatcher-level CUDA graph branch test that constructs a tiny piece_transformer network view and scratch view, captures ring-slot graph templates, and runs one small depth.
- Initial red build command:
  `docker run --rm --gpus all -v D:\100XH100\.worktrees\stream1-piece-transformer:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake --build build-task6-red --target dispatcher_cuda_tests -j2"`
- Expected failure before implementation: `DispatcherNetwork` had no `backend`, `transformer_view`, or `transformer_scratch_lanes` fields.

## Build Verification
- Image: `gpu-dev-cutlass-nsight:cuda128-sm120`.
- Visible local GPU for runnable tests: `NVIDIA GeForce RTX 3070 Laptop GPU, compute_cap=8.6`.
- Required `sm_120` target build command:
  `docker run --rm --gpus all -v D:\100XH100\.worktrees\stream1-piece-transformer:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake -S . -B build-task6 -G Ninja -DCMAKE_BUILD_TYPE=Release -DBEAM_CUDA_ARCHITECTURES=120 && cmake --build build-task6 --target production_runner dispatcher_cuda_tests stream1_transformer_cuda_tests -j2"`
- Result: `production_runner`, `dispatcher_cuda_tests`, and `stream1_transformer_cuda_tests` built for `sm_120`.
- Running `sm_120` binaries on the local `sm_86` GPU failed as expected with `no kernel image is available for execution on the device`; reran execution build for `sm_86`.

## Test Verification
- Runnable build and test command:
  `docker run --rm --gpus all -v D:\100XH100\.worktrees\stream1-piece-transformer:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake -S . -B build-task6-sm86 -G Ninja -DCMAKE_BUILD_TYPE=Release -DBEAM_CUDA_ARCHITECTURES=86 && cmake --build build-task6-sm86 --target production_runner dispatcher_cuda_tests stream1_transformer_cuda_tests stream1_cuda_tests -j2 && ./build-task6-sm86/stream1_transformer_cuda_tests && ./build-task6-sm86/stream1_cuda_tests && ./build-task6-sm86/dispatcher_cuda_tests"`
- Result:
  - `stream1_transformer_cuda_tests=pass`
  - `stream1_cuda_tests=pass`
  - `dispatcher_cuda_tests=pass`
- The local fixture path `test_results/stream1_transformer_reference/weights_fp16` was present, so `stream1_transformer_cuda_tests` exercised the fixture-present reference path.

## Production Smoke
- Transformer model-mode smoke command:
  `docker run --rm --gpus all -v D:\100XH100\.worktrees\stream1-piece-transformer:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "BEAM_RUNTIME_CONFIG_MODE=manual BEAM_WEIGHT_DIR=test_results/stream1_transformer_reference/weights_fp16 BEAM_B_MICRO=4 BEAM_STREAM3_RING_SLOTS=1 BEAM_SHARD_COUNT=1 BEAM_SHARD_BUFFER_COUNT=2 BEAM_SHARD_CAPACITY_CANDIDATES=128 BEAM_STREAM4_BATCH_CANDIDATES=64 BEAM_STREAM4_TRIGGER_CANDIDATES=64 BEAM_STREAM4_ACTIVE_SORT_SLOTS=1 BEAM_STREAM4_BATCH_ALIGNMENT=1 BEAM_GPU_HEADROOM_BYTES=0 ./build-task6-sm86/production_runner 0 1 16"`
- Result:
  - `stream1_mode=model`
  - `stream1_backend=piece_transformer`
  - `stream1_transformer_dims seq_len=51 d_model=256 nhead=8 head_dim=32 layers=4 ff_dim=1024 output_dim=24`
  - `puzzle_solved=0 puzzle_id=0 ...`

- Transformer uniform-mode smoke command:
  `docker run --rm --gpus all -v D:\100XH100\.worktrees\stream1-piece-transformer:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "BEAM_RUNTIME_CONFIG_MODE=manual BEAM_STREAM1_MODE=uniform BEAM_WEIGHT_DIR=test_results/stream1_transformer_reference/weights_fp16 BEAM_B_MICRO=4 BEAM_STREAM3_RING_SLOTS=1 BEAM_SHARD_COUNT=1 BEAM_SHARD_BUFFER_COUNT=2 BEAM_SHARD_CAPACITY_CANDIDATES=128 BEAM_STREAM4_BATCH_CANDIDATES=64 BEAM_STREAM4_TRIGGER_CANDIDATES=64 BEAM_STREAM4_ACTIVE_SORT_SLOTS=1 BEAM_STREAM4_BATCH_ALIGNMENT=1 BEAM_GPU_HEADROOM_BYTES=0 ./build-task6-sm86/production_runner 0 1 16"`
- Result:
  - `stream1_mode=uniform`
  - `stream1_backend=piece_transformer`
  - `stream1_transformer_dims seq_len=51 d_model=256 nhead=8 head_dim=32 layers=4 ff_dim=1024 output_dim=24`
  - `puzzle_solved=0 puzzle_id=0 ...`

## Notes
- A first tiny smoke using auto runtime config and transformer weights failed with `no runtime config fits GPU/final-layout budget`; rerunning with explicit manual sizing passed. This was a configuration search limitation for the deliberately tiny smoke, not a dispatcher wiring failure.
- No cluster jobs were submitted.