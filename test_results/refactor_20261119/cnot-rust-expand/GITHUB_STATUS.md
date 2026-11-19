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
| [44608338814](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44608338814) | Rust/PyO3 validation passed; 226 passed, 2 skipped; NCCL probe syntax check passed | [14572455904](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44608338814/artifacts/14572455904) |

GREEN artifact SHA-256:
`d12b7297f82aa3f1f9fd22190ff01415a47453a4386541273be5e5d0bc360096`.

The current Actions checks run on hosted CPU and verify only the syntax of the
two-rank NCCL probe. Physical CUDA/NCCL multi-GPU execution remains unverified.
Main-branch verification is pending.
