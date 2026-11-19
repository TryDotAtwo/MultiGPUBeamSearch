# CNOT Rust deduplicate/score dispatch

## Change

For typed `PyCnotSearchInput`, Rust now owns the compatibility stage
dispatch, error coordination, and assignment of `children`, `meta`, and
`values`. The CUDA path reuses `distinct_candidates` and the existing model
scorer. CPU and older CUDA libraries without the operator preserve the
`cnot._distinct` fallback. Python-only providers retain the
`CnotStages.deduplicate_and_score` callback.

State identity, candidate order, full-matrix comparison, scoring policy, rank
ownership, collective order and public API are unchanged. This is a port only,
not a search optimization.

## Hosted evidence

| Run | Result | Artifact |
|---|---|---|
| [44611227980](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44611227980) | Setup failure: rustfmt list referenced a source file before it was created; no tests | [14571503063](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44611227980/artifacts/14571503063) |
| [44611473601](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44611473601) | RED as expected: callback rejection only; 2 failed, 226 passed, 2 skipped | [14572841818](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44611473601/artifacts/14572841818) |
| [44611898645](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44611898645) | rustfmt differences only; no compile or tests | [14571313686](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44611898645/artifacts/14571313686) |
| [44612192213](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44612192213) | Full Rust/PyO3 validation passed; 228 passed, 2 skipped; NCCL probe syntax check passed | [14573550966](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44612192213/artifacts/14573550966) |
| [44612790620](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44612790620) | Main after fast-forward; full Rust/PyO3 validation passed; 228 passed, 2 skipped; NCCL probe syntax check passed | [14574490208](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44612790620/artifacts/14574490208) |

GREEN artifact SHA-256:
`720500058c0a0654440c7e15b6aed0c9f019ca1bc417abf236e34095cfd4da20`.

The hosted suite runs on CPU. It checks only syntax for the two-rank NCCL probe;
physical CUDA/NCCL multi-GPU runtime remains unverified. Main-branch
verification is pending.
