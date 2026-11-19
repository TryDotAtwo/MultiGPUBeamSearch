# CNOT Rust expansion dispatch

## Change

For typed `PyCnotSearchInput`, Rust now owns the compatibility stage
dispatch, rank-wide error coordination, and result assignment for expansion.
The CUDA path calls the existing scorer and `expand_candidates` operator.
CPU and CUDA libraries without that optional operator preserve the existing
`cnot._expand` fallback. Python-only and older providers retain the
`CnotStages.expand` callback.

Candidate layout, scoring policy, rank assignment, collective order and public
API are unchanged. No search optimization is included.

## Hosted evidence

| Run | Result | Artifact |
|---|---|---|
| [44607543751](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44607543751) | RED as expected: callback rejection only; 2 failed, 224 passed, 2 skipped | [14570771948](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44607543751/artifacts/14570771948) |
| [44608338814](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44608338814) | CPU/PyO3 tests passed before the new module was added to explicit rustfmt coverage; 226 passed, 2 skipped | [14572455904](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44608338814/artifacts/14572455904) |
| [44609082589](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44609082589) | rustfmt differences only; no compile or tests | [14571746432](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44609082589/artifacts/14571746432) |
| [44609439668](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44609439668) | Remaining rustfmt difference only; no compile or tests | [14571183469](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44609439668/artifacts/14571183469) |
| [44609624223](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44609624223) | Full Rust/PyO3 validation passed; 226 passed, 2 skipped; NCCL probe syntax check passed | [14572517003](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44609624223/artifacts/14572517003) |
| [44610197406](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44610197406) | Main after fast-forward; Rust/PyO3 validation passed; 226 passed, 2 skipped; NCCL probe syntax check passed | [14572542673](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44610197406/artifacts/14572542673) |

Final GREEN artifact SHA-256:
`1fb7af8be19bbc97a8e5d5bdf1d65bdf94017a2d904c557ae26f7424f858ae71`.

The current Actions checks run on hosted CPU and verify only the syntax of the
two-rank NCCL probe. Physical CUDA/NCCL multi-GPU execution remains unverified.
Main-branch verification passed on `1f08dc15c22f692baccd8b732b971c6f023deb65`. Artifact SHA-256: `b6f273918591dec7bc9914061faa3026f8fae1cb7787d7cc70162d6b3290f2bf`.
