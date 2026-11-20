# Typed CPU CNOT child application

## Scope

Move typed Rust compatibility-path CPU CNOT child application into beam-core. Preserve the packed state format, control-major gate IDs, XOR semantics, tail padding, Python result layout, existing CUDA child kernels, Python-only fallback, and multi-GPU protocol.

## Evidence

- RED: [Actions run 44665240841](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44665240841) failed at the expected rejection of the Python child helper: 266 passed, 1 failed, 2 skipped. Artifact [14597487561](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44665240841/artifacts/14597487561), SHA-256 4cd4cc8f2dd63637a751a7ad3d853a468150d07a697e7dff2d2e834ec43cb782.
- GREEN: [Actions run 44669210341](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44669210341) passed formatting, Rust core/runtime/native-sys tests, beam-python check/build, 267 Python CPU compatibility tests (2 skipped), and NCCL probe syntax.
- Hosted artifact: [14598234867](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44669210341/artifacts/14598234867), SHA-256 3969e20eeb4210b5c7b2041d411a3b6571bcd4d01f6745a2a0fef07e4c6c31b6. Full logs remain in GitHub Actions.
- Physical CUDA/NCCL multi-GPU execution: not run. Paid compute: not used.

