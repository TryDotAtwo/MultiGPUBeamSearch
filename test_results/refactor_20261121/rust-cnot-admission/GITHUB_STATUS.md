# Rust + CUDA/C++ migration acceptance status

Source branch: codex/beam-3060-preflight-20261121
Tested source SHA: 4cad49f3c98fe1bc37895af98bfffd5827a0829f

| Gate | Status | Evidence |
|---|---|---|
| Source archive, remote preflight and workflow validation | PASS on hosted Linux CPU fixtures | [run 44946584217](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584217), [artifact 14723403934](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584217/artifacts/14723403934), SHA-256 68a7692500d325d01b683a79564f7ad820ee1ca12178475953eface51998c911 |
| Installed PEP 517 wheel; Rust permutation/CNOT smoke and typed preparation identity v2 | PASS on hosted Linux | [run 44946584220](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584220), [artifact 14724077257](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584220/artifacts/14724077257), SHA-256 58ec47161cfb381e009412e12b116378b5e277fcb54d7ad29e981d0ecf0c9f93 |
| CNOT CPU/Gloo phase, admission, dtype, error coordination and retry suite | PASS: 312 passed, 2 skipped | [run 44946584249](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584249), [artifact 14723688156](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584249/artifacts/14723688156), SHA-256 e1fb89a51013395b43b947a9996f8d68dfbb63311d9b1affe7f52a6beb8868b5 |
| Python resident-session ownership and typed preparation adapter | PASS on hosted Linux | [run 44946584203](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584203), [artifact 14723997251](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584203/artifacts/14723997251), SHA-256 18ab19696753ad16767ba8fef0cb388b28941d48e9b8724cffbc1e50171a1cea |
| LRX(4) B=64 smoke and saturated permutation oracle | CPU acceptance contracts pass on the preceding documented gate; no physical execution | The saturated LRX(5), B=3 fixture checks exact global cutoff, full Hash128 deduplication, cross-rank duplicates, rank-local work and independent replay. Physical traces are still required. |
| CUDA/NCCL permutation and trained CNOT on physical 1/2 GPU | PENDING; GPU job skipped | No Vast instance was dispatched. Active-context UUIDs, physical rank participation, checkpoint scoring, CUDA/NCCL parity and recovery remain unverified. |
| Vast campaign cost, trusted private-input staging and storage disposition | BLOCKED | A trusted GL4 manifest/checkpoint staging receipt and complete storage-cost disposition are still required before the authorized new 2x desktop RTX 3060 campaign. The old instance is outside scope. |
| Full Rust ownership migration | IN PROGRESS | Current gates validate installed CPU behavior and CNOT Gloo control. Remaining permutation preparation/profile and CNOT World/runtime decisions still cross Python ownership boundaries. |

Hosted source and wheel checks do not establish CUDA/NCCL execution. The preparation schema v2 result is a Rust session-contract step, not completion of the full migration.
