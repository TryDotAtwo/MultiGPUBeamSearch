# Rust + CUDA/C++ migration acceptance status

Source branch: `codex/beam-3060-preflight-20261121`
Latest installed-wheel tested code SHA: `565694c013863238bb9ce38e403e35dfd1231221`
Latest Rust execution-plan CPU tested code SHA: `565694c013863238bb9ce38e403e35dfd1231221`
Base main SHA: `e6c41b6dc6bb3a839f6338f3cfe9eaadf8608d5a`

| Gate | Status | Evidence |
|---|---|---|
| Permutation installed wheel and CPU acceptance | PASS | Installed PEP 517 wheel passed on `565694c` in [run 45139332961](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45139332961), including the real Rust execution-plan consumer, exact and max width cases, legacy fallback, live Rust error handling, and existing saturated permutation CPU acceptance. Redacted status artifact: [14801331950](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45139332961/artifacts/14801331950). The public Rust local-rank launcher also passed on `fd42ebc` in [run 45131297173](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45131297173). |
| Shared CPU/Gloo checks | PASS | Rust CPU checks passed on source SHA `565694c` in [run 45139333020](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45139333020), including 3/3 search-plan contract tests; artifact [14800709775](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45139333020/artifacts/14800709775). The installed-wheel run also passed the real Rust execution-plan consumer. This remains CPU/Gloo evidence only. |
| Permutation CUDA/NCCL on physical 1/2 GPUs | PENDING | No physical permutation run was performed. Device participation, 1/2-GPU parity, and hardware replay remain unverified. |
| CNOT implementation and compatibility | PRESERVED; final acceptance DEFERRED | Existing CNOT code and compatibility remain in place. Further CNOT development, checkpoint preparation, and final acceptance are deferred at the user's request; shared CPU workflow success does not establish CNOT GPU acceptance. |
| Full Rust ownership migration | IN PROGRESS | Rust now also supervises single-host local ranks through the public launcher. Python still constructs the graph contract, exports/prepares framework models and runtime, and adapts public results; older wheels retain the Python launcher. PR #10 remains draft. |

Installed-wheel and hosted CPU results do not establish physical CUDA/NCCL execution. The permutation physical gate remains required. CNOT-specific continuation and final acceptance are deferred.

## Rust-owned public local rank launcher — 2026-11-23

- Installed-wheel run [45131297173](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45131297173) passed on source SHA `fd42ebca8d7486c92a7d3a26c98a7ea591cc6f8f`. Its launcher step executed the installed extension outside the checkout, started two synthetic ranks with the public argv/environment contract, and confirmed cleanup of a surviving sibling after an injected rank failure.
- This is hosted CPU supervision evidence. It does not establish CUDA/NCCL execution or physical 1/2-GPU parity.

## Rust-owned permutation solution policy environment — 2026-11-22

- The public permutation runner resolves first/collect mode, extra collection depths, maximum collected solutions, and the TSV result name through the versioned Rust runtime API after calibration.
- The installed-wheel test exercised the Rust extension through the launch-policy helper, checked first/collect values, invalid-mode rejection, and the old-wheel fallback. The installed workflow also passed the existing saturated permutation CPU acceptance.
- Evidence: installed-wheel validation [45062575238](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45062575238), source SHA `c9b6e10b2f013d7fc243719d1d7316faba78b88b`. Physical permutation CUDA/NCCL was not run.

## Rust-owned permutation CUDA device admission — 2026-11-22

- The public permutation preparation path now sends CUDA availability and normalized device inputs to one versioned `beam-runtime` plan. Rust rejects unavailable CUDA and resolves explicit, `num_gpus`, or graph-derived indices. Python retains Linux/torchrun checks, Torch observation, and compatibility with older extensions.
- The installed-wheel test reached the real Rust plan through `backend.runtime_devices`, checked the unavailable-CUDA boundary and the old API fallback, and passed on SHA `e4e96c452dba8b531c7de262f8f333b088a579c3`: [run 45058727951](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45058727951). The same run passed saturated permutation CPU acceptance.
- Shared Rust CPU/Gloo and formatting checks passed: [run 45058728166](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45058728166). The hosted acceptance harness passed on CPU: [run 45058728397](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45058728397); physical jobs were skipped.

## Rust-owned child CUDA visibility translation — 2026-11-22

- `backend.child_environment` now delegates logical selected-index translation through inherited `CUDA_VISIBLE_DEVICES` to the versioned Rust runtime API. It preserves token order, duplicate selections, whitespace trimming, empty/device-disabled/out-of-range errors, `CUDA_DEVICE_ORDER`, environment filtering, and old-wheel fallback.
- Rust CNOT CPU/Gloo passed on SHA `db3e1beb0027b4c4ef42c80b87468fdf3630a07f`: [run 45055128201](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45055128201).
- Installed PEP 517 wheel passed: [run 45055128252](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45055128252). The test ran outside the checkout and exercised the installed PyO3 function through `child_environment`, plus the compatibility fallback and error path.
- Hosted acceptance passed in [run 45055128570](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45055128570); physical and private producer jobs were skipped. An intermediate CPU run caught rustfmt-only differences; they were fixed before the tested SHA.

## Rust-owned scheduler-to-CUDA ordinal mapping — 2026-11-22

- `ScheduledRank.cuda_ordinal` delegates to a separately versioned Rust runtime API. Single-visible-device mapping, local ordinal mapping, error text and old-wheel fallback remain compatible.
- Rust CPU/Gloo validation passed on tested SHA `935016b513a742067f3be11315b7013a322dbefc`: [run 45054019605](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45054019605). It ran the runtime cluster contract tests and Rust formatting checks.
- Installed PEP 517 wheel validation passed outside the checkout: [run 45054019597](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45054019597). Tests called the actual installed extension through the public method, checked single/multi-visible-device behavior, invalid inputs, propagated errors, and the legacy fallback.
- Hosted acceptance passed: [run 45054020205](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45054020205). The physical GPU job was skipped. Intermediate test-fixture and formatting failures were corrected before this tested SHA.

## Rust-owned torchrun/Slurm scheduler rank resolution — 2026-11-22

- The public `cluster.scheduler_rank` helper delegates torchrun and Slurm environment parsing to `beam-runtime`; Python keeps the same `ScheduledRank`, error contract, and fallback for older wheels.
- Test-only RED commit `8565e8b` failed because the installed extension and adapter did not expose `resolve_scheduler_rank`.
- On implementation SHA `a96be795a011174d7f244c06f1593a62b16e6c27`, Rust CNOT CPU/Gloo passed in [run 45052314834](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45052314834), installed-wheel validation passed in [run 45052314835](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45052314835), and the hosted acceptance harness passed in [run 45052315268](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45052315268).
- The latest installed-wheel run `45054019597` also exercises torchrun and Slurm resolution on the updated runtime, with Python parsing disabled for the installed path and compatibility fallback checked separately.
- These are hosted CPU checks; physical CUDA/NCCL remains unverified.

## Rust-owned public cluster planning — 2026-11-22

- The public `plan_cluster` helper now delegates the existing node/rank placement, exact beam capacity, and shard alignment plan to `beam-runtime`. The Python API and result dataclasses are unchanged.
- Rust CPU/Gloo suite passed on `9fa887577d2c1684da8ddb6db29395493ebe45bb`: [run 45050490783](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45050490783). The uneven `(2,4,1)` plan checks every global/node/local rank, requested/effective beam, alignment and local capacity; overflow cases reject before use.
- Installed wheel outside checkout passed: [run 45050490824](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45050490824). It calls `resolve_cluster_plan`, then the public helper with the Python planner forced to fail, and verifies old-extension fallback separately.
- Hosted acceptance harness passed: [run 45050491058](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45050491058); physical GPU acceptance was skipped. No CUDA/NCCL result is claimed.

## Rust-owned permutation microbatch plan — 2026-11-22

- `beam-runtime` now returns the typed plan for the existing inference row/capacity formula. The Python adapter applies the same environment variable and preserves the old formula only for installed extensions without this API.
- Installed-wheel PASS on code SHA `5cfff58de62f454fe6b63d8ea61514718e4ed578`: [run 45048372751](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45048372751). The public helper was exercised with three distinguishable plans, including scalar-head mapping, ensemble with a nondefault row limit, and an unadjusted capacity plan. The installed Rust path was checked with Python fallback disabled; the legacy-extension fallback was checked separately.
- Rust CNOT CPU/Gloo: PASS, [run 45048372758](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45048372758). Hosted acceptance harness: PASS, [run 45048373196](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45048373196). Panel-v2 compatibility: PASS, [run 45048377696](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45048377696).
- Physical CUDA/NCCL job was skipped; no hardware result is claimed.


## Compatibility-route CNOT identity initialization — 2026-11-22

- Test-only installed-wheel SHA `6cd3dc1` exposed a rank disagreement when identity initialization failed on rank 0; the repeat diagnostic at `48e5a32` reported that the two ranks returned different compatibility-dispatch errors.
- Implementation SHA `74c3c33` moves identity creation into a checked Gloo phase before transport startup.
- Installed PEP 517 wheel PASS: [run 45044225072](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45044225072). The same test injects the failure on both rank positions, checks matching phase and cause, confirms no transport startup on the failed call, then replays the healthy public search on the same Gloo group.
- Rust CNOT CPU/Gloo PASS: 323 passed, 2 hardware-only skipped, [run 45044225075](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45044225075).
- Hosted acceptance source harness PASS; physical acceptance skipped, [run 45044225521](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45044225521).

## Latest rank-common dispatch and Gloo parity gate

Tested code SHA: `87e1726e9aad31bae211e7ee07c3ee077729dfc6`.

- Installed PEP 517 wheel outside checkout: PASS, [run 44981755387](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44981755387). All 27 steps completed, including installed-wheel strict/auto/Rust dispatch selection, the two-rank `verify_cnot.main` lifecycle, wide cutoff parity, separate topology deduplication/tie parity, permutation CPU acceptance, and workflow lint. Status artifact [14737644688](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44981755387/artifacts/14737644688).
- Hosted acceptance harness: PASS, [run 44981755539](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44981755539). Panel-v2 CPU compile and synthetic handoff: PASS, [run 44981758527](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44981758527).
- The wide CNOT fixture separately proves a saturated global cutoff; the topology fixture proves cross-parent deduplication and exact full-word ties. One-rank and two-rank winners and layer contracts match.
- Physical CUDA/NCCL acceptance: SKIPPED, [job 137083616404](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44981755539/job/137083616404). No physical 1/2-GPU parity or trained checkpoint execution is claimed. The Vast campaign remains blocked by trusted private-input staging and total storage-cost disposition.


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

## Typed rank-bound CNOT request — 2026-11-22

Tested code SHA: `25bc43abc1911d6a96c14d7c7d0c766a043f5f4d`.

- `CnotRequest` is bound to the active rank topology as `RankedCnotRequest` during collective admission. Invalid or oversized world geometry is rejected before native backend admission.
- Rust CNOT CPU/Gloo: PASS, [run 44980014848](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44980014848).
- Installed PEP 517 wheel: PASS, [run 44980014912](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44980014912), including the two-rank `verify_cnot.main` lifecycle and the corrected receipt contract: rank-local events for both ranks, distinct local engine inputs checked separately, and exactly three solved/bitwise-replayed rank-zero reports. Status artifact [14736474151](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44980014912/artifacts/14736474151).
- Hosted acceptance harness: PASS, [run 44980014854](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44980014854). Panel-v2 CPU compile/handoff: PASS, [run 44980018575](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44980018575).
- Physical CUDA/NCCL: SKIPPED. No physical GPU acceptance occurred.

This is hosted CPU and installed-wheel evidence only. The CNOT controller still receives topology from Python `World`; native context ownership and physical 1/2-GPU parity remain open.



## Latest verified source gate — 2026-11-22

Tested code SHA: `57bf4f0deb90e1b0009b1a458968e5ca4096fe23`.

- Installed PEP 517 wheel outside the checkout: PASS, all 27 steps in [run 44983817156](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44983817156). This includes rank-local two-rank Gloo event receipts, separate engine-input validation, rank-zero-only solved/bitwise replay reports, CNOT cutoff and topology fixtures, and the panel-v2 byte-binding regression.
- Pinned panel-v2 producer CPU compile and synthetic handoff: PASS, [runs 44983817105](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44983817105) and [44983819836](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44983819836).
- Physical CUDA/NCCL: SKIPPED; no instance was dispatched. Installed-wheel/Gloo and panel CPU results do not establish GPU execution, checkpoint scorer parity on hardware, or 1/2-GPU parity.

## Installed scheduler worker and Rust rank supervisor

Tested source SHA: `3241de1df83c101abe3cca906fa9d1de801f611a`.

- Installed PEP 517 wheel ran outside the checkout with `PYTHONPATH` unset and `REQUIRE_RUST_CLUSTER=1`: PASS, [run 44986148911](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44986148911). This proves the wheel contained the Rust cluster extension and worker route.
- The gate exercised both Python and Rust supervisors, both `--local-rank` flag spellings, two-rank shared scheduler environment, Slurm step mapping, coordinated peer failure, timeout, and SIGTERM child cleanup. It did not initialize CUDA/NCCL.
- Hosted acceptance/source harness: PASS, [run 44986148934](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44986148934). Panel-v2 CPU compile and handoff: PASS, [run 44986152290](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44986152290).
- The first source harness attempt at run 44985950759 caught an added raw log upload that violated the workflow's redacted-status-only contract. The log artifact and `tee` were removed; the subsequent source harness passed.
- Physical CUDA/NCCL job: SKIPPED. No paid instance was created; physical 1/2-GPU parity and trained checkpoint execution remain pending.

## Admitted CNOT topology at native-stage binding

Implementation SHA: `d9814462b05938f3419aa5c71332f9c3ce6619cd`; latest hosted test SHA: `519897eef927d7c60603260c2464f53addf8ea6b`.

- Installed PEP 517 wheel outside checkout: PASS, 31 steps, [run 44988105808](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44988105808). The topology failure/recovery fixture passed on two Gloo ranks.
- Rust CNOT CPU/Gloo: PASS after correcting a test expectation for the optional empty admission argument, [run 44988942783](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44988942783).
- Panel-v2 CPU compatibility: PASS, [run 44988946712](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44988946712). Hosted source acceptance passed at implementation SHA [run 44988105821](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44988105821).
- Physical CUDA/NCCL: SKIPPED. No 1/2-GPU behavior is claimed. Python `World` remains the Gloo/tensor-collective owner; full Rust context ownership remains open.


## Single-pass Rust CNOT dispatch — 2026-11-22

Tested source SHA: `9ce5954b6b1d6bd014a2af086dec71ff7686a255`.

- A new installed-wheel regression failed on test-only SHA `edcafd46a1585deb81c4f6007db877015a699fae` at the rank-consistent dispatch step, exposing the redundant depth-selection call in the PyO3 wrapper.
- The wrapper now converts the engine token and depth-fit value once and passes them to the sole route policy, `beam_runtime::cnot_search::select_dispatch`. The exact installed PEP 517 wheel outside-checkout gate passed, including the new selector regression, two-rank dispatch consensus, phase failures and retry, CNOT one/two-rank Gloo parity, and permutation CPU acceptance: [run 44996962384](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44996962384). Redacted status artifact: [14743928675](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44996962384/artifacts/14743928675).
- Rust CNOT CPU/Gloo: 323 passed, 2 hardware-only skipped, [run 44996962379](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44996962379).
- Hosted acceptance: PASS, physical job SKIPPED, [run 44996962394](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44996962394). Panel-v2 CPU compile/handoff: PASS, [run 44996966326](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44996966326).
- No physical CUDA/NCCL or trained-checkpoint GPU execution is claimed.


## Shared profile registry lookup in Rust — 2026-11-22

Tested code SHA: `0ca418cbbc2757508dc96ff986af75b888cba5e2`.

- Installed PEP 517 wheel outside checkout: PASS, including Rust shared-profile lookup, Python adapter authority, mismatch/tamper rejection, legacy-extension fallback, Rust selector, Gloo failure/retry, CNOT 1/2-rank Gloo parity, and permutation CPU acceptance: [run 44999384648](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44999384648), redacted artifact [14745650691](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44999384648/artifacts/14745650691).
- Rust CNOT CPU/Gloo: PASS, 323 passed, 2 hardware-only skipped; shared-profile runtime tests passed, including path escape and symlink rejection: [run 44999384690](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44999384690), log artifact [14745685215](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44999384690/artifacts/14745685215).
- Hosted source acceptance: PASS, [run 44999384719](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44999384719), redacted artifact [14744704817](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44999384719/artifacts/14744704817). Physical acceptance: SKIPPED.
- Pinned panel-v2 CPU compile and synthetic handoff: PASS, [run 44999387783](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44999387783).
- Remaining in this slice: Python still assembles the profile identity; physical 1/2-GPU and trained-profile execution remain unverified.


## Rust-owned permutation calibration and capacity phase plan — 2026-11-23

- The public permutation runner now consumes a versioned `beam-runtime` plan for inference calibration, fixed-width capacity discovery, maximum-frontier refinement, downstream calibration and working-capacity selection. Existing stage code and ordering are retained.
- Rust CPU phase-plan tests passed (2/2) in [run 45135569729](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45135569729), artifact [14799169266](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45135569729/artifacts/14799169266). Installed PEP 517 wheel and the real Rust phase-plan consumer passed on source SHA `258cd9f3428edd2fb89212c92290ac59a520527a` in [run 45135335424](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45135335424); the Rust phase-plan marker is in its hosted job log. Physical permutation CUDA/NCCL on one and two GPUs remains pending. CNOT development, checkpoint preparation and final acceptance remain deferred.


## Rust-owned permutation execution width — 2026-11-23

- Rust CPU execution-plan tests passed 3/3 on source SHA `565694c` in [run 45139333020](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45139333020).
- Installed PEP 517 wheel passed on the same source SHA in [run 45139332961](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45139332961). The runner passed Rust's exact/max width into microbatch preparation; exact width, max width, old-wheel fallback and live Rust resolver failure behavior were covered.
- Physical permutation CUDA/NCCL on one and two GPUs remains pending. The automatic CNOT GPU workflow skipped its physical jobs on the push event; CNOT remains deferred.
