# Rust owns native outcome file persistence

## Source and expected RED

- Test-only commit: [5152324](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/5152324e96a4842277595d4c5f580058cfb63fa3).
- Both hosted RED workflows reached their Python test steps and failed: [Rust public owner CPU validation, run 44495752865](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44495752865) and [CNOT CPU compatibility, run 44495752921](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44495752921).

## Implementation

- Ownership-transfer implementation: [c22dfad](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/c22dfad078881fe7a266c51c2d6e874a58886de0).
- Compatibility follow-up: [d402482](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/d4024829490cb3b096b3236ac01c8d279eaa944d), preserving the existing trailing newline for Python-written fallback results.
- Rust resident sessions defer Python's intermediate outcome-file write only after the Rust worker has returned an outcome and exposes the finalization method. Rust writes the result after the callback supplies prepared metadata. Python sessions, legacy extensions, and incapable Rust owners keep the Python writer and its existing JSON shape.
- The NativeOutcome Python value, search algorithm, CNOT semantics, rank order and GPU stages are unchanged.

## Hosted verification for final source commit

- [Rust public owner CPU validation, run 44500293210](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44500293210) passed the public-owner migration test step. Artifact [14517366049](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44500293210/artifacts/14517366049), SHA-256 8da3b5d44da3f88dca46ecd5ef4372ab1bc074043721c2365c4019aba13e2f80.
- [CNOT CPU compatibility, run 44500293727](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44500293727) passed Rust CNOT checks, the PyO3 integration build, CNOT CPU compatibility tests, and NCCL probe syntax. Artifact [14517386387](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44500293727/artifacts/14517386387), SHA-256 0eee191e8d6e5b396d3d5a5f8bda41559cc3cc2b6739e627f4a8889055b5b2c6.
- Full logs and test outputs remain in those GitHub Actions artifacts. These are hosted CPU checks; no physical CUDA/NCCL multi-GPU acceptance is claimed.
