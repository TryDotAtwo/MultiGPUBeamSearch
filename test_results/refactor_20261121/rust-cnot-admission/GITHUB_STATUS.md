# Rust + CUDA/C++ migration acceptance status

Source branch: `codex/beam-3060-preflight-20261121`
Tested source SHA: `bcd92d0a404802604cabf28f7a477ffe41c8d7de`

| Gate | Status | Evidence |
|---|---|---|
| Source archive, remote preflight and workflow validation | PASS on hosted Linux CPU fixtures | [source run 44943664015](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44943664015), [artifact 14722971129](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44943664015/artifacts/14722971129), SHA-256 `a5fa376eca14d877ddad7b6251e160a7afa42f73c99334c885c15ff0b9e0bbb0` |
| Installed PEP 517 wheel outside checkout; 11 permutation acceptance contracts; strict CNOT admission, dtype, Gloo phases/reuse and public CPU/Gloo search | PASS on hosted Linux | [wheel run 44943664035](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44943664035), [artifact 14722976629](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44943664035/artifacts/14722976629), SHA-256 `5469e291f8379cf26b29490e12c89b33dade15f88158803a5caa3b3d33784613` |
| LRX(4) B=64 public smoke and LRX(5) B=3 saturated CPU oracle | PASS as acceptance logic; GPU job skipped | LRX(5) fixture is saturated, has cross-rank duplicates and identical ordered full Hash128 winners for world sizes 1 and 2. Installed-wheel CPU contracts validate the oracle and trace binding, not the physical runner. |
| CUDA/NCCL permutation and CNOT on physical 1/2 GPU | PENDING; GPU job skipped | No instance was dispatched. Active-context UUIDs, rank participation, trained CNOT checkpoint execution, CUDA/NCCL parity and physical failure recovery remain unverified. |
| Vast campaign cost, trusted private-input staging and storage disposition | BLOCKED | Fresh trusted GL4 manifest/checkpoint staging receipt and full storage-cost disposition are still required before the bounded new 2x desktop RTX 3060 campaign. The old instance is outside scope. |
| Full Rust ownership migration | IN PROGRESS | These gates validate the current installed wheel and acceptance paths; they do not complete the remaining permutation preparation/profile or CNOT World/runtime ownership migration. |

Hosted source and installed-wheel checks establish CPU/Gloo behavior only. They do not establish trained-checkpoint CUDA/NCCL acceptance or completion of the full Rust migration.
