# Rust CNOT strict admission and phase boundaries

Source branch: `codex/beam-3060-preflight-20261121`
Source SHA: `78087abe5c1b8e794c9e16aa934f6afe084c0f8b`

| Gate | Current status | Evidence |
|---|---|---|
| Strict backend admission: ABI/capability and activation agreement, Rust controller/scorer, checkpoint/manifest, DeviceDomain and Stream 5 contract | PASS on hosted CPU Gloo | [run 44891727924](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44891727924), artifact [14701749750](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44891727924/artifacts/14701749750) |
| Dtype contract: int64 states/indices, int64/uint16 move IDs, warning-as-error installed wheel | PASS | [run 44891727924](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44891727924), artifact [14701749750](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44891727924/artifacts/14701749750) |
| Gloo rank-local failures: winner packet and history edge on both ranks, then an independently replayed public search in the same group | PASS | [run 44891727924](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44891727924), artifact [14701749750](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44891727924/artifacts/14701749750) |
| Vast target identity, lifetime cost cap, private manifest receipt, fake SSH partial-result retrieval | PASS for hosted fixtures and source syntax | [run 44891727882](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44891727882), artifact [14701789289](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44891727882/artifacts/14701789289) |
| Physical CUDA/NCCL 1/2 GPU acceptance on RTX 3060 | PENDING; not run | No new instance was dispatched; live private-input staging and physical rank/device evidence remain unverified |

The installed-wheel job also passed public CNOT search and Actionlint. These
results establish CPU/Gloo behavior for the installed wheel and acceptance
harness source checks; they do not establish CUDA, NCCL, trained-checkpoint
execution, or physical 1-/2-GPU parity. No local build/test, paid GPU work, or
result download was used.
