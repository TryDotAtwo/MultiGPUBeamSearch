# Native archive PTX gate regression

- Date: 2026-08-01
- Branch: `codex/megaminx-native-cluster-release`
- Cause: the checker rejected the word `ptx` in the empty `cuobjdump --list-ptx` heading.
- Fix: reject actual PTX file records, `.ptx` payload names, `compute_*` images, and `ptxas`; allow an empty PTX section heading.
- RED: the new empty-heading regression failed with `PTX/JIT image is forbidden` before the parser change.
- GREEN: `tests/portable/test_megaminx_native_release.py` reached `19 passed, 1 skipped`.
- Environment note: the local Windows pytest process did not exit after reporting completion and was terminated by the 120-second command timeout.
- Required final gate: GitHub-hosted CUDA archive build and checker for all six native SM targets.
- Root build fix: use `-DBEAM_CUDA_ARCHITECTURES=${sm}-real`; the unsuffixed CMake architecture emitted virtual PTX by design.
- Workflow contract: `11 passed` before the local pytest teardown timeout.
- Parser refinement: accept `No PTX file found`; reject only numbered `PTX file N:` records and other payload evidence.
- Public-path fix: `CMAKE_SKIP_RPATH=TRUE` plus C++/CUDA `-ffile-prefix-map`; focused workflow test passed.
