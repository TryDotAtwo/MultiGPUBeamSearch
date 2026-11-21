# Rust + CUDA/C++ migration acceptance status

Source branch: `codex/beam-3060-preflight-20261121`
Latest tested code SHA: `77cc69c421a640dc0ec32ee2ce68564b1ded67d3`

| Gate | Status | Evidence |
|---|---|---|
| Hosted source harness and new-host acceptance workflow checks | PASS | [run 44950126040](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44950126040). All hosted CPU checks passed; its physical GPU job was skipped. |
| Installed PEP 517 wheel outside checkout | PASS | [run 44950126048](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44950126048), [artifact 14724934157](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44950126048/artifacts/14724934157), SHA-256 `58f8024b1df53370a5d5292cad9980c34c99e34e88dd8086043612d332e271d9`. Wheel build and CPU/Gloo acceptance passed. |
| CNOT independent scorer-reference cutoff | PASS | Three installed-wheel fixtures passed: exact full-path tie-break, separated reference boundary, and explicit ambiguity within tolerance. The acceptance verifier records boundary status without changing product search policy. |
| CNOT dtype, strict admission, Gloo lifecycle and public Rust-controller search | PASS | Installed-wheel checks passed in run 44950126048. The separate CNOT CPU/Gloo suite passed 312 tests, 2 skipped in [run 44950126166](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44950126166). |
| LRX(4), B=64 smoke and saturated LRX(5), B=3 CPU oracle | PASS on CPU only | Installed-wheel acceptance checked exact cutoff, full Hash128 deduplication, cross-rank duplicates, rank-local work, replay and session reuse. No physical GPU ran. |
| CUDA/NCCL permutation and trained CNOT on physical 1/2 GPUs | PENDING; GPU job skipped | Physical device identity, trained checkpoint scoring, NCCL participation/parity and physical recovery remain unverified. |
| Vast campaign cost, trusted private-input staging and storage disposition | BLOCKED | Trusted GL4 manifest/checkpoint staging receipt and complete storage-cost disposition are still required before the single authorized new 2x desktop RTX 3060 campaign. The old instance is outside scope. |
| Full Rust ownership migration | IN PROGRESS | Typed Rust session identity is in place; permutation model/profile preparation and CNOT World ownership still retain Python control. |

Hosted CPU and installed-wheel checks do not establish physical CUDA/NCCL execution or complete the migration.
