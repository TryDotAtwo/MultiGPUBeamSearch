# Rust CNOT compatibility goal stage

Verified main commit: `53f97cce65185630154c4c09851577bcd58232fe`.

- Corrected test-first run [44399877219](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44399877219) failed on the expected call to the Python goal path (162 passed, 2 skipped).
- Green branch run [44400257858](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44400257858) passed Rust formatting/checks, the PyO3 build, 163 Python tests (2 skipped), and NCCL probe syntax.
- Hosted log artifact [14471674104](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44400257858/artifacts/14471674104), SHA-256 `107089bf02bbb70c4184c339f04e57e8a764f496326a2bb3cdc48ba0c7de818e`.
- Main run [44400879269](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44400879269) passed the same checks: 163 passed, 2 skipped.
- Main artifact [14474065000](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44400879269/artifacts/14474065000), SHA-256 `d7393cf17c75f014ce44c81b676103a668e22344df9bb3dde1929a401a21e758`.

The compatibility stage now delegates goal evaluation to the Rust helper when available. It retains its Python fallback for old extensions and propagates runtime errors without retrying the same collective stage. Search rules and collective order are unchanged.

Physical CUDA/NCCL and multi-GPU CNOT execution remain unverified by this CPU workflow.
