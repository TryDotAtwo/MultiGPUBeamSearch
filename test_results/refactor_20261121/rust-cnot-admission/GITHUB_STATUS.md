# Rust + CUDA/C++ migration acceptance status

Source branch: `codex/beam-3060-preflight-20261121`
Latest tested code SHA: `a517f8afec9de82ad3fe1958fe8b83ba5df2fafa`
Base main SHA: `e6c41b6dc6bb3a839f6338f3cfe9eaadf8608d5a`

| Gate | Status | Evidence |
|---|---|---|
| Hosted source harness and new-host acceptance workflow checks | PASS | [run 44955518309](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44955518309): hosted CPU checks and workflow contracts passed. Physical GPU job skipped. Redacted status artifact [14727009020](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44955518309/artifacts/14727009020). |
| Installed PEP 517 wheel outside checkout | PASS | [run 44955518273](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44955518273): wheel build/install, public CNOT Gloo search, dtype/admission, rank-local failure recovery, CNOT event/reference/pair verifiers, permutation CPU oracle, GPU identity helper, verifier Gloo lifecycle and workflow lint all passed. Redacted status artifact [14726534470](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44955518273/artifacts/14726534470). |
| Default CNOT Rust controller dispatch | PASS on 2-rank Gloo | The public `verify_cnot.main` entry used `engine=auto`; the test replaced the Python reference-loop callback with a failing sentinel. Both healthy searches after rank-0 and rank-1 scorer-load failures completed in the same Gloo group. |
| Vendored source manifest and native fingerprint | PASS | Installed-wheel tests checked required source-bundle headers and that changing vendored headers changes the native fingerprint. |
| CNOT independent scorer-reference cutoff | PASS | Exact full path tie-break, separated boundary and ambiguous-within-tolerance fixtures pass. Search order and precision are unchanged. |
| Permutation acceptance | PASS on CPU only | LRX(4), B=64 smoke and saturated LRX(5), B=3 oracle contracts pass for full Hash128 dedup, duplicate routing, exact cutoff, rank work, replay and session reuse. No physical GPU ran. |
| CUDA/NCCL permutation and trained CNOT on physical 1/2 GPUs | PENDING; GPU job skipped | Physical device identity, trained checkpoint scoring, NCCL participation/parity and physical recovery remain unverified. |
| Vast campaign cost, trusted private-input staging and storage disposition | BLOCKED | A trusted GL4 manifest/checkpoint staging receipt and complete storage-cost disposition are still required before the authorized new 2x desktop RTX 3060 campaign under the remaining $10 cap. The old instance is outside scope. |
| Full Rust ownership migration | IN PROGRESS | Rust controls CNOT stage order/history, but its native stage adapter retains Python `World`/Torch objects. Permutation model/profile preparation and scheduler compatibility also retain Python control. |

These hosted CPU and installed-wheel results do not establish physical CUDA/NCCL execution or complete the Rust migration.
