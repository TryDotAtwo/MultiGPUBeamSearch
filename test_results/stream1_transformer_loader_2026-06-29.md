# Stream1 Transformer Loader Verification - 2026-06-29

## Scope
- Added manifest-only contract tests for `backend=piece_transformer` parsing.
- Added exact-size fake transformer weight directory coverage for exported tensor files.
- Added a real existing MLP `stream1_weights` load smoke to guard MLP loader behavior.
- Added transformer host byte containers, CUDA device pointer containers, upload/free branches, and reusable scratch arena allocation helpers.
- Did not implement transformer CUDA forward and did not wire production runner/dispatcher execution.

## Commands

```powershell
cl /nologo /std:c++20 /EHsc /I src /DBEAM_STATE_LOGICAL_BYTES=120 /DBEAM_STATE_PHYSICAL_BYTES=128 /DBEAM_STATE_ALIGNMENT=16 /DBEAM_MOVE_COUNT=24 tests\contract_tests.cpp src\config.cpp src\frontier_cpu.cpp src\hash.cpp src\state.cpp src\stream3.cpp src\stream4.cpp /Fe:test_results\contract_tests_stream1_task4_red.exe
```

Result: initial red route not run because plain `cl` was not on PATH in this PowerShell environment (`cl` not recognized). Visual Studio Build Tools were then located via `vswhere`.

```powershell
cmd /c "\"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat\" -arch=x64 >nul && cl /nologo /std:c++20 /EHsc /I src /DBEAM_STATE_LOGICAL_BYTES=120 /DBEAM_STATE_PHYSICAL_BYTES=128 /DBEAM_STATE_ALIGNMENT=16 /DBEAM_MOVE_COUNT=24 tests\contract_tests.cpp src\config.cpp src\frontier_cpu.cpp src\hash.cpp src\state.cpp src\stream3.cpp src\stream4.cpp /Fe:test_results\contract_tests_stream1_task4.exe"
```

Result: PASS, executable built.

```powershell
.\test_results\contract_tests_stream1_task4.exe
```

Result: PASS, `contract_tests=pass`. Report updated at `test_results/contract_tests_2026-05-20.md` with transformer manifest, transformer exact-size loader, and MLP loading checks.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; py -B -m unittest discover -s tests -p "test_stream1_transformer_exporter.py" -v
```

Result: PASS, 6 tests.

```powershell
cmake -S . -B build-contract-tests
```

Result: BLOCKED. Configure stopped while probing Visual Studio `VCTargetsPath` because MSBuild could not delete its generated `unsuccessfulbuild` file under `build-contract-tests` (`Access is denied`). Full CMake/CUDA target verification was not available from this shell.

```powershell
git diff --check -- tests/contract_tests.cpp tools/stream1_weight_io.hpp
```

Result: PASS. Git warned that CRLF working-copy files will be normalized to LF when touched.

## Cleanup Note
- MSVC/CMake generated root `.obj` files, `$null`, and `build-contract-tests/`. Both PowerShell `Remove-Item` and `cmd del/rmdir` were denied by the Windows filesystem for those generated artifacts, so they remain untracked and were not staged.

## Task 4 Quality Fix Follow-up

### Changes Verified
- Added regression coverage that malformed `transformer_layers` does not fall back to `num_layers`.
- Added regression coverage that manifests containing both `transformer_layers` and `num_layers` with different values are rejected.
- Added fail-closed `piece_transformer Stream1 forward is not wired yet` guards in `production_runner` and `stream_benchmark` before transformer runtime sizing or GPU allocation paths.
- Added catch/free/rethrow cleanup around `upload_weights()` and `alloc_stream1_scratch()` partial CUDA allocations.
- Recorded the later-task FlashAttention 2 / compatible-transformer forward requirement in `memory/PROMPTS.md`; no transformer forward kernel or dispatcher was implemented.

### Red Check

```powershell
& $env:ComSpec /c '"C:\Program Files (x86)\Microsoft Visual Studio2\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul && cl /nologo /std:c++20 /EHsc /I src /DBEAM_STATE_LOGICAL_BYTES=120 /DBEAM_STATE_PHYSICAL_BYTES=128 /DBEAM_STATE_ALIGNMENT=16 /DBEAM_MOVE_COUNT=24 tests\contract_tests.cpp src\config.cpp srcrontier_cpu.cpp src\hash.cpp src\state.cpp src\stream3.cpp src\stream4.cpp /Fe:test_results\contract_tests_stream1_task4_quality_red.exe'
.	est_results\contract_tests_stream1_task4_quality_red.exe
```

Result: FAIL before loader fix, `contract_tests=fail error=malformed transformer_layers must not fall back to num_layers`.

### Final Verification

```powershell
& $env:ComSpec /c '"C:\Program Files (x86)\Microsoft Visual Studio2\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul && cl /nologo /std:c++20 /EHsc /I src /DBEAM_STATE_LOGICAL_BYTES=120 /DBEAM_STATE_PHYSICAL_BYTES=128 /DBEAM_STATE_ALIGNMENT=16 /DBEAM_MOVE_COUNT=24 tests\contract_tests.cpp src\config.cpp srcrontier_cpu.cpp src\hash.cpp src\state.cpp src\stream3.cpp src\stream4.cpp /Fe:test_results\contract_tests_stream1_task4_quality.exe'
.	est_results\contract_tests_stream1_task4_quality.exe
```

Result: PASS, `contract_tests=pass`.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; py -B -m unittest discover -s tests -p "test_stream1_transformer_exporter.py" -v
```

Result: PASS, 6 tests.


```powershell
git diff --check
```

Result: PASS. Git reported CRLF normalization warnings for touched working-copy files only; no whitespace errors.
