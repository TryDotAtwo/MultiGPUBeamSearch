# Rust + CUDA/C++ migration acceptance status

Source branch: `codex/beam-3060-preflight-20261121`
Latest tested code SHA: `c59d6294c11c411c9c1eaad54115b689b00a10a3`
Base main SHA: `e6c41b6dc6bb3a839f6338f3cfe9eaadf8608d5a`

| Gate | Status | Evidence |
|---|---|---|
| Hosted source harness and new-host acceptance workflow checks | PASS | [run 44956473161](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44956473161): hosted source CPU checks and workflow contracts passed. Physical GPU job skipped. Redacted status artifact [14727716957](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44956473161/artifacts/14727716957). |
| Rust CNOT CPU compatibility suite | PASS | [run 44956473123](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44956473123): Rust formatting/checks and CNOT CPU/Gloo compatibility tests passed on this SHA. |
| Installed PEP 517 wheel outside checkout | PASS | [run 44956473162](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44956473162): wheel build/install, public CNOT Gloo search, dtype/admission, rank-local failure recovery, CNOT event/reference/pair verifiers, permutation CPU oracle, GPU identity helper, verifier lifecycle and workflow lint all passed. Redacted status artifact [14727464522](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44956473162/artifacts/14727464522). |
| Default CNOT Rust controller dispatch | PASS on 2-rank Gloo | Public `verify_cnot.main` used `engine=auto` with the Python reference-loop callback replaced by a failing sentinel. Both healthy searches after rank-0 and rank-1 scorer-load failures completed in the same Gloo group. |
| Runtime engine policy | PASS | `beam-runtime::cnot_search::select_engine` is exercised for auto/Rust/Python choices and oversized native depth. |
| Vendored source manifest and native fingerprint | PASS | Installed-wheel tests checked required source-bundle headers and that changing vendored headers changes the native fingerprint. |
| CNOT independent scorer-reference cutoff | PASS | Exact full path tie-break, separated boundary and ambiguous-within-tolerance fixtures pass. Search order and precision are unchanged. |
| Permutation acceptance | PASS on CPU only | LRX(4), B=64 smoke and saturated LRX(5), B=3 oracle contracts pass for full Hash128 dedup, duplicate routing, exact cutoff, rank work, replay and session reuse. No physical GPU ran. |
| CUDA/NCCL permutation and trained CNOT on physical 1/2 GPUs | PENDING; GPU job skipped | Physical device identity, trained checkpoint scoring, NCCL participation/parity and physical recovery remain unverified. |
| Vast campaign cost, trusted private-input staging and storage disposition | BLOCKED | A trusted GL4 manifest/checkpoint staging receipt and complete storage-cost disposition are still required before the authorized new 2x desktop RTX 3060 campaign under the remaining $10 cap. The old instance is outside scope. |
| Full Rust ownership migration | IN PROGRESS | Rust controls CNOT stage order/history and engine policy, but the native stage adapter retains Python `World`/Torch objects. Permutation model/profile preparation and scheduler compatibility also retain Python control. |

Hosted CPU and installed-wheel evidence does not establish physical CUDA/NCCL execution or complete the Rust migration.

- Hosted source harness [run 44969306210](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44969306210) passed; physical GPU job skipped. Panel-v2 CPU source compile and synthetic handoff [run 44969309554](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44969309554) passed. No physical CUDA/NCCL execution occurred.


## Latest installed Gloo lifecycle gate

Tested code SHA: `caec731140d21452e40072b056962133e2cb82ca` (status entry is documentation-only).

- Hosted source harness: PASS, [run 44973855889](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44973855889). The physical acceptance job was skipped; no GPU instance was dispatched.
- Installed wheel: PASS, [run 44973855896](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44973855896). The two-rank `verify_cnot.main` Gloo lifecycle covered mismatched per-rank engine requests, rank-0 and rank-1 injected trained-scorer failures, and a successful healthy search after each failure. Each rank wrote its own phase receipts; only rank zero wrote the three replay reports required by the validator.
- Panel-v2 producer compatibility on CPU: PASS, [run 44973858891](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44973858891).
- CUDA/NCCL execution, physical GPU identity, trained-checkpoint GPU scoring, and 1/2-GPU parity: NOT RUN; the physical job remains skipped and the Vast campaign is paused.


## Typed CNOT request owned by beam-runtime, 2026-11-22

The public Rust CNOT admission now creates one typed `beam_runtime::cnot_search::CnotRequest` after the existing collective validation vote. It carries validated wires, global width, alphabet-capped shortlist, depth, mode, audit policy, validation policy, and requested engine; the admitted request supplies the values used to build the Rust-controlled search. Existing tensor snapshot, scoring, stage order, and collective sequence are unchanged.

On source SHA `0c43e75fc6dcf69457b920923682ff3d34b5a41f`, the Rust CNOT CPU/Gloo suite passed in [run 44975670295](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44975670295), the installed wheel passed in [run 44975670268](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44975670268), the hosted acceptance harness passed in [run 44975670365](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44975670365), and the panel-v2 CPU compile/handoff gate passed in [run 44975674616](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44975674616). The physical GPU job was skipped. These checks establish hosted CPU and installed-wheel behavior only; CUDA/NCCL, trained checkpoint execution on hardware, and 1/2-GPU parity remain pending.


## Typed CNOT request conversion to runtime search options

Tested code SHA: `c63937944b7752227982b388ee075be26e89c055`.

- Rust CNOT CPU/Gloo: PASS, [run 44976811410](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44976811410).
- Installed PEP 517 wheel: PASS, [run 44976811495](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44976811495).
- Hosted source acceptance harness: PASS, [run 44976811467](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44976811467).
- Panel-v2 CPU compile and synthetic handoff: PASS, [run 44976815210](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44976815210).
- Physical CUDA/NCCL job: SKIPPED; no GPU was dispatched.

The request now supplies normalized controls to the generic Rust compatibility-stage controller; the public PyO3 `run_cnot` signature and search semantics remain unchanged. This is hosted CPU evidence, not hardware acceptance.


## CNOT verify_cnot Gloo receipts, 2026-11-22

Tested code SHA: `c59d6294c11c411c9c1eaad54115b689b00a10a3`.

- The two-rank test records each phase in rank-local event receipts. The engine-mismatch case allows the expected local inputs to differ (`auto` and `python`) while checking the common rejection outcome.
- The verifier checks exactly three rank-0 result reports, all `solved` with `bitwise_replay=true`: `healthy-after-engine-mismatch-0.json`, `healthy-after-0-0.json`, and `healthy-after-1-0.json`. It does not expect result JSON from rank 1.
- Hosted installed-wheel Gloo lifecycle and receipt validation: PASS, [run 44977925758](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44977925758). Receipt contract CPU fixture includes missing-phase, wrong-rank and missing-root-replay rejection.
- Hosted acceptance harness: PASS, [run 44977925762](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44977925762). Rust CNOT CPU/Gloo: PASS, [run 44977925871](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44977925871). Panel-v2 CPU compile/handoff: PASS, [run 44977931023](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44977931023).
- Physical CUDA/NCCL job: SKIPPED. No Vast instance was dispatched.
