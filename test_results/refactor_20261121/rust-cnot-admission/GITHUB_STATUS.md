# Rust + CUDA/C++ migration acceptance status

Source branch: `codex/beam-3060-preflight-20261121`
Tested source SHA: `b682c6ffea3ab7ab24ce0d36fe4fcabee59cc505`

| Gate | Status | Evidence |
|---|---|---|
| Vast campaign identity, marker ordering, bounded stop/readback and prior-label restoration | PASS on hosted Linux CPU fixtures; no live Vast API call | [source run 44931235596](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44931235596), [artifact 14717802816](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44931235596/artifacts/14717802816) |
| Installed wheel build, public CNOT search, strict backend admission, dtype and two-rank Gloo failure/reuse | PASS on hosted Linux CPU/Gloo | [wheel run 44931235590](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44931235590), [artifact 14717507673](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44931235590/artifacts/14717507673) |
| CUDA/NCCL permutation and CNOT on physical 1/2 GPU | PENDING; GPU job skipped | No GPU instance was rented or dispatched. Physical backend, rank participation, saturation, replay and error recovery remain unverified. |
| Hard $10 total-cost bound | BLOCKED | Vast storage continues billing after stop. The current 24-hour reserve is not a lifetime cap; no data deletion was authorized or performed. |

The hosted checks establish source/CPU behavior and an installed CPU wheel only. They do not establish trained-checkpoint execution or physical CUDA/NCCL parity.
