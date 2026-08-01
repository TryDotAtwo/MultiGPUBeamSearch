# Stream1 Transformer Weight Estimate Contract 2026-06-29

Scope: Fix P3 quality-review finding for piece-transformer `fast_slot_projected` sizing. Runtime budget estimates and dispatcher fixture allocation now match the loader/exporter contract: `max_piece_size * num_classes * d_model` half values.

Red check:
- Command: `docker run --rm --gpus all -v "D:\100XH100\.worktrees\stream1-piece-transformer:/workspace" -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake --build build-task6-sm86 --target dispatcher_cuda_tests -j2 && ./build-task6-sm86/dispatcher_cuda_tests"`
- Result before runtime fix: FAIL as expected with `piece_transformer runtime weight estimate must size fast_slot_projected by max_piece_size, not state_len`.

Final verification:
- Command: `docker run --rm --gpus all -v "D:\100XH100\.worktrees\stream1-piece-transformer:/workspace" -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake --build build-task6-sm86 --target dispatcher_cuda_tests -j2 && ./build-task6-sm86/dispatcher_cuda_tests"`
- Result: PASS, `dispatcher_cuda_tests=pass`.