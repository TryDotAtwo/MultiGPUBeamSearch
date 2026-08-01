# Stream1 Piece Transformer Task 1 Verification - 2026-06-29

## Scope
- Added tagged Stream1 backend config fields and row-mode contract coverage.
- Added manifest backend parsing coverage for legacy MLP, explicit MLP, piece_transformer rejection, and unknown backend rejection.

## Commands
- `cmake --build build-gpu-dev-cutlass --target contract_tests`
  - Result: not run; `build-gpu-dev-cutlass` does not exist in this worktree.
- `cmake -S . -B build-contract-tests`
  - Result: configure blocked by missing local CUDA toolset (`No CUDA toolset found`).
- `cl /nologo /std:c++20 /EHsc /I src /DBEAM_STATE_LOGICAL_BYTES=120 /DBEAM_STATE_PHYSICAL_BYTES=128 /DBEAM_STATE_ALIGNMENT=16 /DBEAM_MOVE_COUNT=24 /c tests\contract_tests.cpp`
  - Red result before initial implementation: failed because `Stream1ModelConfig::backend`, transformer metadata fields, and backend constants were undeclared.
- `cl /nologo /std:c++20 /EHsc /I src /DBEAM_STATE_LOGICAL_BYTES=120 /DBEAM_STATE_PHYSICAL_BYTES=128 /DBEAM_STATE_ALIGNMENT=16 /DBEAM_MOVE_COUNT=24 tests\contract_tests.cpp src\config.cpp src\frontier_cpu.cpp src\hash.cpp src\state.cpp src\stream3.cpp src\stream4.cpp /Fe:test_results\contract_tests_stream1_task1.exe`
  - Result: build passed.
- `.\test_results\contract_tests_stream1_task1.exe`
  - Output: `contract_tests=pass`.

## Notes
- Full CMake contract target was not verified because this Windows environment has no CUDA toolset for the project-level `project(... CUDA ...)` configuration.
- The CPU contract executable was built and run directly with MSVC as the smallest available static/runtime check for the touched config and manifest loader contracts.