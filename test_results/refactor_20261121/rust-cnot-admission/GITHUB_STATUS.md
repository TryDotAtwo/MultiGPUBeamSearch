# Rust + CUDA/C++ migration acceptance status

Source branch: `codex/beam-3060-preflight-20261121`
Latest tested source SHA: `9e6863387861c0a643f0a043acfdbe6892126c13`

| Gate | Status | Evidence |
|---|---|---|
| Source archive, remote preflight and workflow validation | PASS on prior docs-only SHA; not rerun after Python adapter-only change | [run 44947270384](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44947270384). This gate covers hosted CPU fixtures, not GPU execution. |
| Installed PEP 517 wheel outside checkout; Rust session reuse and public permutation/CNOT CPU acceptance | PASS on latest tested SHA | [run 44947770665](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44947770665), [artifact 14724367144](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44947770665/artifacts/14724367144), SHA-256 `9a2cd384509198463d78cc99df2ea0ee5ac7e834fc2aee404be8c4ebe42c27af`. Wheel build, installed smoke, backend admission, phase boundaries, public search, event verification and CPU permutation acceptance passed. |
| Python-owner session reuse and nested structured options identity | PASS on same source SHA | [run 44947770658](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44947770658), [artifact 14724530178](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44947770658/artifacts/14724530178), SHA-256 `74681e4acf1d526dead977f441b6b9d1cc69722819b70cfb185b59c59bf535b3`. |
| CNOT CPU/Gloo phase, admission, dtype, error coordination and retry suite | PASS on SHA `4cad49f3c98fe1bc37895af98bfffd5827a0829f`: 312 passed, 2 skipped | [run 44946584249](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584249), [artifact 14723688156](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584249/artifacts/14723688156). Not triggered by the later Python session identity change. |
| LRX(4) B=64 smoke and saturated permutation oracle | CPU acceptance passed on preceding documented gate; no physical execution | The saturated LRX(5), B=3 fixture checks exact global cutoff, full Hash128 deduplication, cross-rank duplicates, rank-local work and independent replay. |
| CUDA/NCCL permutation and trained CNOT on physical 1/2 GPU | PENDING; GPU job skipped | Active-context UUIDs, physical rank participation, checkpoint scoring, CUDA/NCCL parity and recovery remain unverified. |
| Vast campaign cost, trusted private-input staging and storage disposition | BLOCKED | Trusted GL4 manifest/checkpoint staging receipt and complete storage-cost disposition remain required before the authorized new 2x desktop RTX 3060 campaign. The old instance is outside scope. |
| Full Rust ownership migration | IN PROGRESS | Session options identity now compares typed JSON. Permutation model/runtime profile preparation and CNOT World ownership still retain Python control. |

Hosted source and wheel checks do not establish CUDA/NCCL execution or complete the migration.
