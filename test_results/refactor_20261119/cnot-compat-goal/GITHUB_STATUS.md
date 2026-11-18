# Rust CNOT compatibility goal stage

Verified branch commit: `f398074569de661e507ce9b5cc0d602daa7e0b23`.

- Corrected test-first run [44399877219](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44399877219) failed on the expected call to the Python goal path (162 passed, 2 skipped).
- Green branch run [44400257858](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44400257858) passed Rust formatting/checks, the PyO3 build, 163 Python tests (2 skipped), and NCCL probe syntax.
- Hosted log artifact [14471674104](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44400257858/artifacts/14471674104), SHA-256 `107089bf02bbb70c4184c339f04e57e8a764f496326a2bb3cdc48ba0c7de818e`.

The compatibility stage now delegates goal evaluation to the Rust helper when available. It retains its Python fallback for old extensions and propagates runtime errors without retrying the same collective stage. Search rules and collective order are unchanged.

Physical CUDA/NCCL and multi-GPU CNOT execution remain unverified by this CPU workflow.
