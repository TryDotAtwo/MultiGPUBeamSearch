# Rust + CUDA/C++ migration acceptance status

Source branch: `codex/beam-3060-preflight-20261121`
Latest tested code SHA: `a4994a5a394f7d7f370b7ddc04eb08e175fc8367`
Base main SHA: `e6c41b6dc6bb3a839f6338f3cfe9eaadf8608d5a`

| Gate | Status | Evidence |
|---|---|---|
| Hosted source harness and new-host acceptance workflow checks | PASS | [run 44954310185](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44954310185): all source CPU and workflow-contract steps passed. Physical GPU job skipped. Redacted status artifact [14726792929](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44954310185/artifacts/14726792929). |
| Installed PEP 517 wheel outside checkout | PASS | [run 44954310147](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44954310147): wheel build/install, public CNOT Gloo search, dtype, strict admission, rank-local failure/reuse, CNOT event/reference/pair verifiers, permutation CPU oracle, GPU identity helper, verify_cnot Gloo lifecycle, and workflow lint all passed. Redacted status artifact [14726867983](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44954310147/artifacts/14726867983). |
| Vendored source manifest and native fingerprint | PASS | Same installed-wheel run tested required source bundle headers and invalidation when vendored header content changes. |
| CNOT independent scorer-reference cutoff | PASS | Exact path tie-break, separated boundary, and ambiguous-within-tolerance fixtures pass. Search policy and precision are unchanged. |
| Permutation acceptance | PASS on CPU only | LRX(4), B=64 smoke and saturated LRX(5), B=3 oracle contracts pass, including full Hash128 dedup, cross-rank duplicate handling, exact cutoff, rank work, replay and session reuse. No physical GPU ran. |
| CUDA/NCCL permutation and trained CNOT on physical 1/2 GPUs | PENDING; GPU job skipped | Physical device identity, trained checkpoint scoring, NCCL participation/parity and physical recovery remain unverified. |
| Vast campaign cost, trusted private-input staging and storage disposition | BLOCKED | Trusted GL4 manifest/checkpoint staging receipt and complete storage-cost disposition are still required before the one authorized new 2x desktop RTX 3060 campaign under the remaining $10 cap. The old instance is outside scope. |
| Full Rust ownership migration | IN PROGRESS | Rust controls CNOT stage order/history, but its native stage adapter still retains Python `World`/Torch objects. Permutation model/profile preparation and scheduler compatibility also retain Python control. |

These are hosted CPU/installed-wheel results. They do not establish physical CUDA/NCCL execution, or complete the Rust migration.
