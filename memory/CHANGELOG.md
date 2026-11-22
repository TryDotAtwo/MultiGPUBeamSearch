## 2026-11-22 - Require rank-common CNOT stage bindings

- Fixed Rust dispatch admission so the native controller is selected only when every rank exposes the complete required native-stage set. When that set is absent, `auto` selects the agreed compatibility route; explicit Rust reports a common error before transport if no Rust route is available.
- Added a RED unit regression for a rank missing a native-stage binding and an installed-wheel two-rank Gloo regression for the same missing symbol on rank 0 and rank 1. The regression checks strict rejection before the data plane, the common auto route, explicit Rust rejection when no shared Rust route exists, and a healthy auto retry on the same process group.
- On tested source SHA `87e1726e9aad31bae211e7ee07c3ee077729dfc6`, the installed PEP 517 wheel passed [run 44981755387](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44981755387), including CNOT rank-consistent dispatch and one/two-rank Gloo cutoff, topology deduplication and tie fixtures. The hosted acceptance harness passed [run 44981755539](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44981755539); panel-v2 CPU compile/handoff passed [run 44981758527](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44981758527). The physical CUDA/NCCL job was skipped.
- Search order, scoring, kernels, precision and public API are unchanged. Hardware execution remains unverified.

## 2026-11-22 - Bind CNOT requests to rank geometry

- Added `RankedCnotRequest` in `beam-runtime`. The validated request is bound to the active `rank/world` during the existing collective input-admission phase; invalid topology (`world=0`, `rank >= world`, or `world > 65536`) is rejected before native backend admission. Private fields and accessors preserve the validated contract.
- The Rust CNOT adapter now uses the bound request for normalized search controls. Public Python signatures, tensors, stage ordering, collectives, scoring, kernels, precision and search semantics are unchanged.
- On source SHA `25bc43abc1911d6a96c14d7c7d0c766a043f5f4d`, Rust CNOT CPU/Gloo passed in [run 44980014848](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44980014848), the installed PEP 517 wheel passed in [run 44980014912](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44980014912), hosted acceptance passed in [run 44980014854](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44980014854), and the panel-v2 CPU compile/handoff passed in [run 44980018575](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44980018575). The physical CUDA/NCCL job was skipped.


## 2026-11-22 - Validate CNOT lifecycle receipts by rank and phase

- The installed-wheel lifecycle contract uses one event receipt from each rank for every phase. Engine mismatch receipts retain each rank's different requested engine and validate that value separately while requiring the same rejection phase and outcome.
- `verify_cnot.py` writes its global replay report only on rank 0, so validation now consumes exactly three rank-0 reports: `healthy-after-engine-mismatch-0.json`, `healthy-after-0-0.json`, and `healthy-after-1-0.json`. Rank 1 is not expected to emit a result report.
- The earlier `9a49382` check incorrectly expected a report from each rank. At tested code SHA `c59d6294c11c411c9c1eaad54115b689b00a10a3`, the hosted harness passed [run 44977925762](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44977925762), the installed wheel and two-rank Gloo lifecycle passed [run 44977925758](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44977925758), the Rust CNOT CPU/Gloo suite passed [run 44977925871](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44977925871), and the panel-v2 CPU handoff passed [run 44977931023](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44977931023). The physical GPU job was skipped.

## 2026-11-22 - Build runtime options from the typed CNOT request

- Added `CnotRequest::search_options` in `beam-runtime`. It carries the validated width, alphabet-capped shortlist, derived directed-gate alphabet and depth into the generic Rust CNOT search controller.
- The compatibility-stage branch now converts the admitted request directly to runtime options. The existing `run_cnot` PyO3 call signature remains available through a thin adapter. Search ordering, scorer calls, tensors, kernels and collective sequence are unchanged.
- Hosted RED at `c383aa9` failed on the new missing `search_options` method as expected. On source SHA `c63937944b7752227982b388ee075be26e89c055`, Rust CNOT CPU/Gloo passed in [run 44976811410](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44976811410), installed wheel in [run 44976811495](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44976811495), hosted source harness in [run 44976811467](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44976811467), and panel-v2 CPU compile/handoff in [run 44976815210](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44976815210). The physical GPU job was skipped.

## 2026-11-22 - Move public CNOT route selection into Rust runtime

- Added a typed capability matrix and dispatch decision in `beam-runtime` for the Rust controller, compatibility Rust controller and Python reference path.
- The public CNOT adapter now asks the installed PyO3 runtime to select the route from rank-agreed capabilities. Strict acceptance fails closed unless Rust controller, native stages, Gloo vote and required Stream 5 operations are present on every rank. Older extensions keep the existing non-strict fallback.
- Search ordering, scoring, kernels, precision and result formats are unchanged. Hosted validation is pending.

## 2026-11-22 - Resolve CNOT controller choice in beam-runtime

- Moved the `auto/python/rust` decision table from the PyO3 adapter into typed `beam-runtime::cnot_search` policy. The adapter retains public value normalization and delegates the decision; oversized depth still selects the reference path for `auto` and `python`, while explicit Rust fails with the same error.
- Added Rust unit coverage for all selection branches. The installed-wheel two-rank Gloo verifier now runs its healthy retries with `engine=auto` and fails if the Python reference-loop callback is entered; rank-local scorer failures remain explicitly on Rust.
- Final same-SHA hosted checks passed at `0606965a210c9cc2417b729433471e3d5ca6988c`: CNOT CPU/Gloo [run 44956473123](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44956473123), installed PEP 517 wheel [run 44956473162](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44956473162), source acceptance harness [run 44956473161](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44956473161). Physical GPU work was skipped.
- Search kernels, score ordering, precision and public API did not change. Full CNOT context ownership and physical CUDA/NCCL verification remain open.

## 2026-11-22 - Verify default CNOT Rust controller dispatch

- Extended the installed-wheel `verify_cnot.main` Gloo lifecycle check: explicit `engine=rust` remains on injected scorer failures, while healthy retries use public `engine=auto` with the Python reference-loop callback replaced by a failing sentinel.
- This verifies the installed wheel's default controller selection across rank-0 and rank-1 failures and two subsequent healthy searches in the same two-rank Gloo group. Search, scorer, kernels, precision and numerical policy are unchanged.
- Hosted installed-wheel run [44955518273](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44955518273) passed on tested source SHA `a517f8afec9de82ad3fe1958fe8b83ba5df2fafa`; redacted status artifact [14726534470](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44955518273/artifacts/14726534470). Hosted source harness [44955518309](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44955518309) passed; its physical GPU job was skipped. This does not establish CUDA/NCCL execution.

## 2026-11-22 - Keep watchdog detail reports out of public logs

- Removed raw Vast watchdog report output from the GitHub Actions log. The workflow continues to publish only its allowlisted summary and redacted status artifact.
- Updated the hosted workflow contract fixture for the current run label and corrected newline expectations in the result-transport tests.
- Hosted CPU source validation is pending for this change.

## 2026-11-22 - Compare CNOT cutoff with independent scorer reference

- Extended the CNOT acceptance verifier to record an independent scorer-reference ordering at each layer cutoff. Reports mark underfilled, exact full-path ties, reference-separated cutoffs and boundaries ambiguous within score tolerance; reported-score selection is asserted against the reference only when the boundary is determinate.
- Search order, scorer, precision, kernels and product numerical policy are unchanged.
- On source SHA `77cc69c421a640dc0ec32ee2ce68564b1ded67d3`, the installed PEP 517 wheel passed all three exact-tie, ambiguous-boundary and separated-boundary fixtures in [run 44950126048](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44950126048), [artifact 14724934157](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44950126048/artifacts/14724934157), SHA-256 `58f8024b1df53370a5d5292cad9980c34c99e34e88dd8086043612d332e271d9`. Hosted source harness passed [run 44950126040](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44950126040). The physical CUDA/NCCL job was skipped.
- CNOT CPU/Gloo suite passed 312 tests with 2 skipped on the same source in [run 44950126166](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44950126166).

## 2026-11-21 - Use typed options in resident Rust session identity

- Removed Python repr from the nested identity passed to Rust resident workers. The nested Python session identity now carries the same structured options descriptor as the Rust preparation identity.
- Hosted RED on `97441f6` demonstrated the missing field for world sizes 1 and 2. On `9e6863387861c0a643f0a043acfdbe6892126c13`, Python-owner passed [run 44947770658](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44947770658), artifact [14724530178](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44947770658/artifacts/14724530178); the installed wheel and CPU acceptance passed [run 44947770665](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44947770665), artifact [14724367144](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44947770665/artifacts/14724367144).
- CUDA/NCCL and physical multi-GPU acceptance were not run.

## 2026-11-21 - Store Rust preparation options as typed JSON

- Preparation identity schema v2 carries NativeOptions as a JSON object. Rust validates the object and compares preparation identities structurally; the max beam-width form remains typed alongside numeric widths.
- The Python adapter sends the new schema only to an extension that declares v2. An older extension stays on the compatibility path. Preparation changes continue to invalidate resident state; matching requests can reuse the worker group.
- Hosted validation passed on source SHA 4cad49f3c98fe1bc37895af98bfffd5827a0829f: source harness [44946584217](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584217), artifact [14723403934](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584217/artifacts/14723403934); installed wheel [44946584220](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584220), artifact [14724077257](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584220/artifacts/14724077257); CNOT CPU/Gloo, 312 passed and 2 skipped [44946584249](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584249), artifact [14723688156](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584249/artifacts/14723688156); Python session-owner gate [44946584203](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584203), artifact [14723997251](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44946584203/artifacts/14723997251).
- No physical GPU job ran. This change does not complete migration of all search control into Rust.

## 2026-11-21 - Add acceptance-only saturated permutation GPU evidence

- Added an opt-in native build identity for the permutation acceptance trace. The trace binds run/case/call, source and installed-wheel hashes, native build and binary hashes, rank/PID, CUDA UUID observed from the active context, shard count, and a bounded post-cutoff frontier snapshot.
- Added a deterministic LRX(5) width-3 CPU oracle and installed-wheel verifier path alongside the existing LRX(4), width-64 smoke. The saturated path runs two calls in one Rust session for each of one and two GPUs and checks exact global survivors and independent replay. No search kernel or numerical policy changed.
- Hosted RED on source SHA `38cbae6` (`44941059326`) found that the opt-in CMake flag and replay oracle wiring were incomplete. LRX(4) has no saturated tied-score fixture with topology-parity winners, so the CPU test first tried LRX(5); installed-wheel run `44942970688` confirmed a deterministic parity witness there. A test-only RED on `ac12956` (`44943334188`) caught that the GPU verifier still used LRX(4); `bcd92d0` switches only the saturated case to LRX(5), keeps LRX(4) B=64 smoke, and passes the hosted acceptance checks. The CNOT source fixture missing manifest-required inputs was fixed in `3aad58b`.
- Hosted source harness [run 44943664015](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44943664015), artifact [14722971129](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44943664015/artifacts/14722971129), passed. Installed-wheel run [44943664035](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44943664035), artifact [14722976629](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44943664035/artifacts/14722976629), passed 11 permutation acceptance contracts, strict CNOT gates and public CPU/Gloo search. The GPU job was skipped.
- Physical 1/2-GPU validation has not run. Vast dispatch stays gated on the previously documented cost and staging checks.

## 2026-11-21 - Add saturated permutation acceptance contracts

- Implemented an independent CPU oracle for the existing Hash128 owner/shard order, full-hash dedup, equal-score tie cutoff and contiguous rank partition. The hosted gate now checks a saturated width-3 LRX(5) fixture, cross-rank duplicate states, run/call-bound rank traces and zero padding. LRX(4), B=64 remains the separate public smoke.

## 2026-11-21 - Bind campaign shutdown to the selected Vast instance

- The campaign marker now records the exact instance ID, Actions run and attempt, spend cap, and prior Vast label. The main workflow verifies a fresh account listing before enabling cleanup.
- A separate scheduled and workflow-completion watchdog stops only a marked, ID-matching 2x RTX 3060 campaign and confirms both stopped state and prior-label restoration with fresh API reads.
- Hosted source harness `44931235596` and installed-wheel run `44931235590` passed; the physical GPU job was skipped. The CPU fixtures cover marker identity, API stop/readback, label restoration, and workflow ordering. The watchdog preserves instance data; stopped storage remains billable. Physical dispatch remains gated until the total storage-cost disposition is resolved.

## 2026-11-21 - Preflight CayleyPy, NCCL and pinned CUTLASS before build

- The private GPU-host preflight now requires the trusted CUTLASS archive and extracted header under the pin used by `sources.py`, and emits a separately verified revision/SHA receipt.
- The actual GPU acceptance worker checks the CayleyPy API, NCCL header/library and pinned CUTLASS cache with `setup_sources(offline=True)` before starting the installed-wheel build. It rechecks the same cache through the installed package, so the smoke cannot download CUTLASS lazily.
- Hosted tests cover missing, duplicate and wrong CUTLASS receipts; source and workflow checks validate receipt parsing and pre-build ordering. Physical cache availability and GPU build remain unverified until a 2x RTX 3060 run.

## 2026-11-21 - Reserve the full Vast campaign budget

- The instance contract now requires the fresh storage and bandwidth rates, bounds both source and result archives, subtracts their cost plus 24 hours of stopped storage, and reserves phase wall time, SSH setup, TERM/KILL grace, retrieval and stop time before computing a runtime deadline.
- Each hosted phase obtains its timeout from the same absolute deadline. The private preflight, source staging and upload retain future execution, result collection and stop reserves.
- Hosted verification is required before physical dispatch. Vast storage billing continues after stop; the current 24-hour storage reserve is only an estimate and does not cap longer retention.

## 2026-11-21 - Retain admitted CNOT backend handles through search startup

- Strict CNOT admission now prepares one live DeviceDomain through a rank-wide
  Gloo vote, then carries its handle and SHA-256 into the Rust stages. Stream 5
  carries its fingerprint and live transport through the same admission object;
  the compatibility path does not add library hashing.
- Device dispatch tracing now only emits its local receipt. The existing stage
  vote reports any trace failure, and a new installed-wheel regression injects
  pre-trace failure on either Gloo rank before a healthy retry. DeviceDomain
  preparation failure/retry uses the same production preparation helper.
- Stream 5 setup and admitted-handle binding now coordinate local preparation,
  transport construction and receipt failures before peers advance to another
  collective. `run_cnot_native` keeps its previous Python call shape by making
  the admission handle optional.
- P0 remote payload harness is green on source SHA `ae5d3acd3fd4a809b40e2f25600e02224bbcd5bc`:
  [Actions run 44904838667](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44904838667).
  The first P1 installed wheel on source SHA `29aee4dc9f5161f65a7b338da0bf4f665b2642bf`
  passed all 15 build, install, strict-admission, Gloo recovery, verifier and
  public-search steps in
  [Actions run 44906873707](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44906873707).
  Its source harness passed in
  [run 44906873808](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44906873808);
  the GPU job was skipped. The CNOT compatibility workflow on that SHA found
  two stale event expectations around the newly voted Stream 5 preparation and
  optional admission argument; the follow-up keeps the old call shape and
  updates the vote-order expectation. That compatibility rerun is pending.
  CUDA/NCCL remains unrun and no Vast instance was started.

## 2026-11-21 - Bind paired CNOT receipts to rank identities

- `cnot_acceptance_events.validate_case` now returns the observed per-rank PID,
  process device and physical GPU UUID identities. The paired campaign verifier
  checks those identities against each result report, enforces a distinct UUID
  for the two-rank case, and rejects the legacy Vast instance ID.
- Added a standalone hosted CPU fixture for 1/2-GPU reports, shared invocation
  binding, ordered survivors, missing rank identity, duplicate GPU UUID and
  foreign case/instance inputs. Wired it into both source and installed-wheel
  acceptance workflows. Hosted result is pending.

## 2026-11-21 - Separate PyO3 extension linking from Rust fixture tests

- `beam-python` keeps `extension-module` as its default feature so wheel builds
  preserve the existing PyO3 module behavior. The Rust event serialization
  fixture test disables that feature so its executable links against Python
  instead of leaving extension-module symbols unresolved.
- RED: installed-wheel workflow run
  [44897715956](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44897715956)
  built and installed the wheel, then the Rust serializer test failed at link
  time with unresolved `PyBytes_*`/`PyErr_*` symbols under `extension-module`.
- On source SHA `85e6ac1ed76505caf91a172ac45d7cba99a6bdc9`, the Rust serializer,
  dtype checks, backend admission and phase-boundary checks passed in the
  installed-wheel job. Its actual `verify_cnot.main` lifecycle test then failed
  when it tried to reinitialize torchrun's Gloo rendezvous store after each
  invocation. The test now holds one process group live across rank-local
  scorer failures and healthy retries.
- CNOT CPU compatibility passed on the same SHA:
  [run 44898116700](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44898116700).
- GREEN on `98a2025c1fcb8efb942ce70570c7992a15f0df5c`: installed-wheel run
  [44898482925](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44898482925)
  passed wheel build/install, Rust/Python JSONL serialization, dtype checks,
  strict backend admission, Gloo phase-failure recovery, the actual two-rank
  `verify_cnot.main` scorer-failure/retry scenarios, public CNOT search and
  workflow lint. CNOT CPU compatibility passed in
  [run 44898482934](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44898482934).
  The acceptance source harness passed in
  [run 44898482922](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44898482922);
  its GPU job was skipped. These results do not establish CUDA/NCCL hardware
  acceptance. The paired campaign verifier's new standalone CPU gate is pending.

## 2026-11-21 - Make CNOT runtime acceptance select Rust explicitly

- `tools/verify_cnot.py` accepts `--engine`, reports the selected engine and scorer runtime type, and `--require-rust-scorer` requires both the Rust controller and the PyO3 Rust-owned TQP scorer for trained checkpoint cases. The search algorithm and scorer numerics are unchanged.
- Added an end-to-end CPU/Gloo run of the public CNOT verifier to CNOT compatibility tests and the installed-wheel workflow. The wheel run imports the package from site-packages outside the checkout and uses `engine=rust`; it solved the n=144 fixture in two gates and passed independent bitwise replay.
- RED: [Actions run 44819995307](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44819995307) caught both missing CLI contracts (2 failed, 308 passed, 2 skipped).
- GREEN: [CNOT CPU run 44820597906](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44820597906) passed Rust checks, PyO3 build, 310 Python tests (2 skipped, 104 warnings), and NCCL probe syntax. Artifact [14672625098](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44820597906/artifacts/14672625098), SHA-256 `1e520abb300f25bb14e80384832dda91f79582e1530030ec5941d1625e40425b`.
- [Installed-wheel run 44820639095](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44820639095) built and installed the PEP 517 wheel outside the checkout, passed `pip check`, then ran the Rust CNOT controller through that installed wheel with CPU/Gloo. The n=144, B=1, K=3 case returned length 2 and bitwise replay passed. Artifact [14671043639](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44820639095/artifacts/14671043639), SHA-256 `f0335a4828b4d416234b5d32ba36cbf4b1b2517bbd48f2f65b85a27c45a9f4b1`.
- The squash merge to main, `ecfc4d4ad6084a72ecb34a9a958c3161b143d9c8`, passed again: [CNOT CPU run 44821882474](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44821882474) (310 passed, 2 skipped, 104 warnings), artifact [14672636499](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44821882474/artifacts/14672636499), SHA-256 `37697d7e3dac3e097e79a89c1d4d6a3ee685ac5cf6b5d663bb0ee8d452decc50`; and [installed-wheel run 44821882308](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44821882308), artifact [14672282878](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44821882308/artifacts/14672282878), SHA-256 `816f1d22a449a47d8b995374eafc4c25b639e5a01fc6fb1427cc0c4130214b9a`.
- These runs do not load the private checkpoint and do not verify CUDA/NCCL or physical 1-/2-GPU execution. Full receipt: `test_results/refactor_20261121/cnot-native-verifier/GITHUB_STATUS.md`.

## 2026-11-20 - CNOT setup failures use the Gloo control group

- The Rust top-level CNOT controller now coordinates failures from identity
  construction, native stage runner loading, and compatibility stage
  construction with the existing checked CNOT protocol before peers continue.
  Native-stage capability negotiation uses `World.all_ranks_support` over
  CPU/Gloo rather than the search payload transport.
- Added two-rank Gloo failure injection for all three setup operations. Each
  case checks the same rank-wide diagnostic and confirms the process group
  remains usable. Updated capability-vote tests to require the Gloo control
  method.
- Search order, scoring, precision, owner mapping, and NCCL payload exchange
  were not changed.
- RED: Actions run
  [44750014338](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44750014338)
  failed the new assertions on the pre-fix behavior (4 failed, 296 passed,
  2 skipped). Follow-up
  [44751261363](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44751261363)
  passed Rust checks and PyO3 build but exposed three outdated test expectations.
- GREEN: Actions run
  [44751923980](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44751923980)
  passed Rust CNOT checks, PyO3 build, 300 Python tests (2 skipped), and NCCL
  probe syntax. Artifact [14636575322](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44751923980/artifacts/14636575322),
  SHA-256 `d69aa55acf31b468732a5ce30dfb0318ddab15d9bb07eda06e21922e70fc3f19`.
- Physical CUDA/NCCL and 1-/2-GPU execution remain unverified.
- Main commit `7f48cea985fad82176411d1731eb7d4d741bced7` passed again:
  [CNOT CPU run 44753288626](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44753288626)
  passed Rust checks, PyO3 build, 300 Python tests (2 skipped), and NCCL probe
  syntax. Artifact [14635354179](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44753288626/artifacts/14635354179),
  SHA-256 `ad6f1a99d901646fb5e1be98c52bfdaedbdd223ee6b7b24e9b8364d1fdb05844`.
- The same main commit passed the installed PEP 517 wheel check:
  [run 44753288676](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44753288676)
  built and installed the wheel outside the source tree, passed `pip check`,
  and verified public exports plus `resident-search-v1`. Artifact
  [14636965521](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44753288676/artifacts/14636965521),
  SHA-256 `ce53ecc010af096c80fdde650488d602eb729de792f050c0d36e7698b46ecdc9`.
- Existing-Vast status run
  [44753288462](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44753288462)
  confirmed instance 61080396 is exited and skipped CUDA/NCCL execution; it made
  no start, stop, or rent request. Artifact
  [14635019938](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44753288462/artifacts/14635019938),
  SHA-256 `148f76a3806e2bd0943e825ea9c6a740f4c87ffbcd2a26d5476d553e442fde68`.
- Full receipt:
  `test_results/refactor_20261120/cnot-setup-gloo/GITHUB_STATUS.md`.

## 2026-11-20 - Align Rust CNOT stages with the Gloo checked-control ABI

- The Rust compatibility dispatcher now recognizes the new checked-control capability, so typed CNOT initialize, expand, and deduplicate/score stages no longer fall back because they still looked for the old v2 marker. The top-level two-rank controller test now asserts a Gloo failure vote rather than the former payload-world vote.
- RED: [Actions run 44744338479](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44744338479) exposed 12 compatibility failures. [Run 44745437675](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44745437675) reduced these to one stale event-order expectation.
- GREEN: [Actions run 44746148112](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746148112) passed Rust CNOT checks, the PyO3 build, the CNOT CPU compatibility suite (297 passed, 2 skipped, 104 warnings), and NCCL probe syntax. Artifact [14632339297](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746148112/artifacts/14632339297), SHA-256 `db25f1da847170b2bf06d64736966023bbb2fff28a3e39bce14e60ae8ee11f26`.
- Main commit `d7fd23dda968ae47cb2346f2c96e2c26b3d2b75c` passed [CNOT CPU run 44746884936](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746884936): 297 passed, 2 skipped, 104 warnings; artifact [14632634476](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746884936/artifacts/14632634476), SHA-256 `596a1c081a1e8f48b0a692d50a6a16d50a4a8fb8170b407ad0a73f269ddb4dd2`.
- [Installed wheel run 44746884980](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746884980) built and installed the PEP 517 wheel, passed `pip check` and external-checkout smoke tests for exports, Rust ABI, AutoMax identity and resident-session reuse. Artifact [14633756534](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746884980/artifacts/14633756534), SHA-256 `23973b113206b82b9e9bbe32ee286088a8324b13621eee7e00e97b00a8801089`. The session test used a fixture worker, not a GPU model.
- [Vast status run 44746884926](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746884926) found instance 61080396 exited and skipped physical execution; it made no start, stop, or rent request. Artifact [14633197432](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44746884926/artifacts/14633197432), SHA-256 `f3858af2cb34b6b6f2c7c38f12109403e78bc793b7668ca8a81c562790ea5cc9`.
- Physical CUDA/NCCL execution and rank-wide runtime recovery remain unverified. Details: `test_results/refactor_20261120/cnot-checked-gloo-control/GITHUB_STATUS.md`.

## 2026-11-20 - Coordinate Rust session identity and CNOT exchange failures

- AutoMax remains a distinct requested preparation value while resolved session identity stores the numeric width. Older PyO3 extensions keep the Python-owned preparation cache.
- CNOT compatibility ranks agree on the requested engine over the Gloo control group before engine-dependent branching. Rust and Python Stream 5 paths now vote on receive-buffer readiness before sending payload; any local allocation failure prevents every rank from entering the payload exchange.
- CPU/Gloo verification: final CNOT workflow at [run 44738874159](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44738874159) passed 294 tests, skipped 2, and reported 104 warnings. The focused Rust `cnot_exchange` integration suite passed 6 tests. Artifact [14628794155](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44738874159/artifacts/14628794155), SHA-256 `a16b41766901ffd35e1a32c9d2ca565ed6db5d7fe544244402707837cb3dd9c1`.
- Installed wheel verification passed at [run 44738874151](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44738874151). Artifact [14629797227](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44738874151/artifacts/14629797227), SHA-256 `80f6cc4467937e301bf42ad1ebcd51fa38ee7bc535217d8cb3b09e779df4cc36`.
- [Vast status run 44738618500](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44738618500) found instance 61080396 exited and skipped the physical check. It made no start, stop, or rent request. CUDA/NCCL runtime acceptance remains open.
- Detailed run and artifact references: `test_results/refactor_20261120/cnot-runtime-failure-safety/GITHUB_STATUS.md`.

## 2026-11-20 - Enter World for CNOT validation collectives

- CNOT validation helpers and the direct rank-exchange fixture now enter `World` before rank-coordinated checks. Capability tests use the actual `World` negotiation interface. Search and exchange algorithms are unchanged.
- RED: [Actions run 44714626102](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44714626102) failed with 4 tests, alongside 274 passed and 2 skipped. Two helper tests supplied a mock without the new capability API; two real-Gloo helpers used `World` outside its entered context. Artifact [14618627216](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44714626102/artifacts/14618627216), SHA-256 `976e92b81ffb09e27915228e1c6f2ae397e8c51664fee0648a2972bcddf59533`.
- GREEN: [Actions run 44715949615](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44715949615) at `8e494c5b0ac0cdedf44c21eb2cc459c466132a2b` passed Rust CNOT checks, PyO3 build, 278 Python tests (2 skipped, 104 warnings), and NCCL probe syntax. Artifact [14618694452](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44715949615/artifacts/14618694452), SHA-256 `6b570f42d4244585b5abc4e015b79494daacab19b663d37a75e8ae07848a346d`.
- This is hosted CPU/Gloo and PyO3-build evidence. It does not verify physical CUDA/NCCL. The earlier status-only Vast workflow [44714626948](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44714626948) found the configured instance exited and skipped checkout/build/GPU execution.

## 2026-11-20 - Coordinate CNOT BaseException failures across ranks

- Multi-rank checked CNOT stages now capture Python `BaseException` values, including `KeyboardInterrupt`, into the existing rank-wide error vote. The Rust checked ABI follows the same rule; single-rank calls preserve the original exception.
- CNOT CPU Actions run [44708561397](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44708561397) passed Rust formatting/checks, Rust CNOT tests, PyO3 build, and 271 Python compatibility tests (2 skipped). Artifact [14615563947](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44708561397/artifacts/14615563947), SHA-256 `5294dd904cc054b5bcff621de6f291f84b77bff1d875aaf50c9e3b145a5f7e17`.
- The existing-Vast workflow [44708561389](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44708561389) authenticated its redacted Actions secret but found instance 61080396 exited and skipped build/transfer/GPU execution. Status artifact [14616430543](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44708561389/artifacts/14616430543), SHA-256 `727c95f07cb1c931629e0cb748aff1e839570dc71993d9efad11e47d63996ac8`.
- CPU ABI and mocked two-rank error tests do not establish physical NCCL or CUDA recovery. Evidence: `test_results/refactor_20261120/cnot-checked-baseexception/GITHUB_STATUS.md`.

## 2026-11-20 - Preserve max beam width across resident sessions

- Resident identity now keeps the requested form of beam width, including `"max"`, while retaining the accepted logical width separately. Repeated resident requests validate against that accepted width.
- Hosted Actions run [44707125022](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44707125022) passed: 86 CPU/Python tests, 3 skipped. Artifact [14614783619](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44707125022/artifacts/14614783619), SHA-256 `afbf5615341c6260dddc4254f1d8638c06026682dd6f24ef1dd9ff4b67bf03b6`.
- This verifies public dispatch and the resident-session adapter on hosted CPU only. It does not compile the Rust extension or validate CUDA/NCCL. Evidence: `test_results/refactor_20261120/auto-max-beam-session/GITHUB_STATUS.md`.

## 2026-11-20 - Add CNOT descriptor recovery to Vast validation

- The existing-instance workflow now builds the standalone CNOT owned-domain ABI and runs the two-rank Stream 5 control and corrected descriptor-error recovery checks. It does not start, stop, or rent the Vast instance.
- Actions run [44705007455](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44705007455) completed successfully. The secret-backed read-only Vast status query worked, but the configured instance was stopped, so checkout, build, and GPU execution were skipped. Sanitized artifact [14614466680](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44705007455/artifacts/14614466680), SHA-256 `84ec4991f053a6914ae815526ab09af29c07f15723cec5135cf7db3e0131d512`.
- This validates workflow parsing and the stopped-instance gate only. CMake, CUDA, NCCL, descriptor-error propagation, and same-process-group recovery remain physically unverified. Evidence: `test_results/refactor_20261120/cnot-device-error-workflow/GITHUB_STATUS.md`.

## 2026-11-20 - Move typed CPU CNOT owner assignment into Rust

- For typed CPU CNOT compatibility searches, owner assignment now uses `beam_core::cnot::owner_ranks` through the PyO3 adapter. The existing CUDA owner operator and Python-only fallback remain in place. Hashes continue to select shards only; state equality and full packed-word comparison are unchanged.
- RED: hosted run [44674560289](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44674560289) failed at the expected Python owner-helper callback check (268 passed, 1 failed, 2 skipped). Artifact 14602160860, SHA-256 `76726ff8bb49e3d56cf7acbf19476c4b6759d6ad552df3b24e3efc9b9a203508`.
- An intermediate hosted run [44698622193](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44698622193) stopped at rustfmt before compilation; formatting was corrected.
- GREEN: [Actions run 44699020897](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44699020897) passed Rust CNOT checks, the PyO3 build, 269 CPU/PyO3 tests (2 skipped), and the NCCL probe syntax check. Artifact 14612107333, SHA-256 `f367eb4217389be60ddaff4e8812565a0a26ecde879962e3ba8bb0e159d92ceb`.
- This is hosted CPU/PyO3 validation only. No physical CUDA/NCCL multi-GPU run or paid compute was used. Evidence: `test_results/refactor_20261120/cnot-rust-owners/GITHUB_STATUS.md`.

# Changelog

## 2026-11-20 - Move typed CPU CNOT first-unseen into Rust
- MAIN GREEN: [Actions run 44672860884](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44672860884) at commit `7a2e679f66d903b6bd309e00006078f1f2617c1c` passed Rust CNOT checks, PyO3 extension build, 268 Python CPU compatibility tests (2 skipped, 98 warnings), and NCCL probe syntax. Artifact [14599974834](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44672860884/artifacts/14599974834), SHA-256 `a7eb4010f90efde25bdc3e057e76f73ca3447fa9f345b5688c4330d4191610ca`.

- The test-first RED run [44671005907](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671005907) reached the typed CPU search and failed exactly because it called the Python cnot_first_unseen tensor helper (267 passed, 1 failed, 2 skipped). Artifact [14600765521](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671005907/artifacts/14600765521), SHA-256 09684db96085d6ee9bef6b6e3b7802c023c5c061ebfc54090f04387f8b939409.
- beam-core now returns the first original child index for each state absent from closed and earlier children. It masks non-logical tail bits and confirms hash-bucket matches by comparing every packed word. Typed CPU PyO3 dispatch calls this core function; CUDA and Python-only paths remain unchanged.
- Hosted attempt [44671574865](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671574865) stopped at rustfmt on one line-wrapping difference in the PyO3 adapter; no Rust tests or build ran. Formatting was corrected. Artifact [14599897789](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671574865/artifacts/14599897789), SHA-256 4b81553d8e059c868d9b7fd7b86db5cdc52b5031c908eeb2acb968aa1bd5216f.
- Hosted attempt [44671760449](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671760449) stopped at rustfmt on one blank line and the public helper signature; no compile/tests ran. Both formatting differences were corrected. Artifact [14600586872](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671760449/artifacts/14600586872), SHA-256 9ad89211797baeafa74bc08201f7cfa2675aa3811da92cfde7ab23819322d482.
- GREEN hosted run [44671946779](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671946779) passed rustfmt, beam-core/runtime/native-sys Rust checks, beam-python cargo check and PyO3 build, 268 Python CPU compatibility tests (2 skipped), and NCCL probe syntax. Artifact [14600462371](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671946779/artifacts/14600462371), SHA-256 92eaa60b94e5a1ca1e96f90f38e12385186db6a801ee28a83e759a67b1675db6.
- Physical CUDA/NCCL multi-GPU execution has not run; paid compute was not used.
- Evidence: test_results/refactor_20261120/cnot-rust-first-unseen/GITHUB_STATUS.md.

## 2026-11-20 - Move typed CPU CNOT child application into Rust

- Added beam_core::cnot::apply_gate_words for in-place application of one physical CNOT to packed row-major state, retaining control-major move IDs and clearing per-row tail padding. The typed CPU compatibility adapter applies selected children through this Rust primitive and returns the existing int64 tensor layout.
- The existing CUDA child kernels and Python-only fallback remain unchanged. Rust core coverage checks non-word-aligned geometries and invalid inputs; hosted CPU/PyO3 coverage rejects the Python tensor child helper.
- Test-first RED run [44665240841](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44665240841) failed at the expected callback-rejection assertion (266 passed, 1 failed, 2 skipped). Artifact [14597487561](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44665240841/artifacts/14597487561), SHA-256 4cd4cc8f2dd63637a751a7ad3d853a468150d07a697e7dff2d2e834ec43cb782.
- Hosted attempt [44667127333](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44667127333) stopped at the parser because the inserted expand signature missed its opening parenthesis; no Rust tests or PyO3 build ran. The signature is corrected. Hosted attempt [44667418609](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44667418609) then stopped at rustfmt on three line-wrap differences; those were corrected for the next hosted run. Hosted attempt [44667758194](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44667758194) passed formatting and core/runtime tests, then cargo check rejected signed i64::div_ceil as unstable on the pinned toolchain. The bounded wire count now uses integer division with an added 63. Hosted attempt [44668029009](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44668029009) passed Rust checks and PyO3 build; the new test matched Python output but its final replay assertion expected gates even though collection was disabled. The unsupported assertion was removed. Hosted attempt [44668846301](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44668846301) passed Rust checks and PyO3 build; Python/reference and Rust results matched, but an extra test assertion incorrectly required status=solved while both returned depth_limit for this fixture. Removed the unnecessary status assertion. No CUDA, NCCL, multi-GPU, or paid compute has run for this change.
- GREEN hosted run [44669210341](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44669210341) passed rustfmt, beam-core/runtime/native-sys Rust checks, beam-python cargo check and PyO3 build, 267 Python CPU compatibility tests (2 skipped), and NCCL probe syntax. Artifact [14598234867](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44669210341/artifacts/14598234867), SHA-256 3969e20eeb4210b5c7b2041d411a3b6571bcd4d01f6745a2a0fef07e4c6c31b6.
- MAIN verification run [44669629830](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44669629830) also passed the same checks: 267 CPU/PyO3 tests passed, 2 skipped. Artifact [14598629346](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44669629830/artifacts/14598629346), SHA-256 f539dc627bec39c0ad9f4efb9dbade15e8f10e68e301bb2bd2aa188ecdbeaa19.
- Physical CUDA/NCCL multi-GPU execution was not run; no paid compute was used.
- Evidence: test_results/refactor_20261120/cnot-rust-cpu-children/GITHUB_STATUS.md.

## 2026-11-20 - Move permutation search-input serialization into Rust

- beam-runtime::search_inputs now writes the legacy permutation puzzle_info.json and test.csv from a Rust-validated PublicGraphContract, preserving byte formatting and generator order. PyO3 rebuilds and validates the normalized request; Python retains the writer for older extensions and legacy contract shapes.
- The test-first RED run [44637299070](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44637299070) reached Rust compilation and failed on the expected missing beam_runtime::search_inputs module. The first compatibility run exposed seven test fixtures without Rust graph fields; the adapter now returns to the existing Python writer for those shapes.
- GREEN CNOT CPU/PyO3 run [44640784456](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44640784456) passed Rust formatting/checks, PyO3 build, 240 Python tests (2 skipped), and NCCL probe syntax. Artifact [14586960403](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44640784456/artifacts/14586960403), SHA-256 81c2e270527d8e2180c8a49f8dfcc3ef627a9bd2b62cbe7fb71ee64fd61a211b. Companion permutation depth-control run [44640371591](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44640371591) passed.
- This transfers task-file serialization only. Search semantics, multi-GPU rank protocol, inference kernels, and CNOT behavior are unchanged. The checks do not include physical CUDA/NCCL multi-GPU execution. Evidence: test_results/refactor_20261120/rust-search-input-serialization/GITHUB_STATUS.md.

## 2026-11-19 - Move CNOT deduplicate/score dispatch into Rust

- For typed `PyCnotSearchInput`, the Rust compatibility adapter now owns distinct-state dispatch, score validation, rank-wide error coordination, and result assignment. It reuses the existing CUDA `distinct_candidates` operator and scorer; CPU and older operator libraries retain `cnot._distinct`.
- RED hosted run [44611473601](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44611473601) failed only the new callback-rejection cases (2 failed, 226 passed, 2 skipped; artifact 14572841818). Run [44611227980](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44611227980) stopped before tests because its rustfmt list named the not-yet-created source module (artifact 14571503063). Formatting-only run [44611898645](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44611898645) reported rustfmt differences before compile/tests (artifact 14571313686).
- Corrected hosted run [44612192213](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44612192213) passed Rust formatting/checks, PyO3 build, and CPU/PyO3 compatibility tests (228 passed, 2 skipped); the NCCL probe passed syntax checking. Artifact [14573550966](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44612192213/artifacts/14573550966), SHA-256 `720500058c0a0654440c7e15b6aed0c9f019ca1bc417abf236e34095cfd4da20`.
- MAIN verification after fast-forward: hosted run [44612790620](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44612790620) passed Rust formatting/checks, PyO3 build, 228 CPU/PyO3 tests (2 skipped), and NCCL probe syntax checking. Artifact [14574490208](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44612790620/artifacts/14574490208), SHA-256 `6aae1edd75e92331c8198c23a466179514e3a6a837adb3920b87236ccaec607c`.
- No search, deduplication, scoring, or transport policy changed. Physical CUDA/NCCL multi-GPU runtime acceptance remains unverified. Evidence: `test_results/refactor_20261119/cnot-rust-dedup-score/GITHUB_STATUS.md`.

## 2026-11-19 - Move CNOT expansion dispatch into Rust

- For typed `PyCnotSearchInput`, the Rust compatibility adapter now owns expansion dispatch, result assignment, and rank-wide error coordination. The CUDA path uses the existing parent scorer and `expand_candidates` operator; CPU and older operator libraries keep the existing `cnot._expand` fallback. Python-only providers retain the stage callback.
- RED hosted run [44607543751](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44607543751) failed only the callback-rejection regression cases (2 failed, 224 passed, 2 skipped; artifact 14570771948). The initial green run [44608338814](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44608338814) passed PyO3 and 226 CPU/PyO3 tests (2 skipped) before the new Rust source was added to the explicit rustfmt list. The two follow-up formatting checks [44609082589](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44609082589) and [44609439668](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44609439668) stopped on rustfmt differences only (artifacts 14571746432 and 14571183469; no compile or tests). Corrected full run [44609624223](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44609624223) passed Rust formatting/checks, PyO3 build and CPU/PyO3 tests (226 passed, 2 skipped), plus NCCL probe syntax checking. Artifact [14572517003](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44609624223/artifacts/14572517003), SHA-256 `1fb7af8be19bbc97a8e5d5bdf1d65bdf94017a2d904c557ae26f7424f858ae71`.
- MAIN verification after fast-forward: hosted run [44610197406](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44610197406) passed Rust formatting/checks, PyO3 build and CPU/PyO3 tests (226 passed, 2 skipped), plus NCCL probe syntax checking. Artifact [14572542673](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44610197406/artifacts/14572542673), SHA-256 `b6f273918591dec7bc9914061faa3026f8fae1cb7787d7cc70162d6b3290f2bf`.
- This changes stage ownership only; candidate bytes, scoring, rank assignment and collective order remain the same. Physical CUDA/NCCL and multi-GPU runtime execution remain unverified. Evidence: `test_results/refactor_20261119/cnot-rust-expand/GITHUB_STATUS.md`.

## 2026-11-19 - Move CNOT initialization dispatch into Rust

- For typed `PyCnotSearchInput`, the compatibility Rust stage adapter now creates the initial parent, `lex`, and `closed` values directly, using the existing CNOT owner operator. Rank-wide error negotiation uses the existing v2/legacy checker semantics; unsupported Python stage providers retain the callback fallback.
- RED hosted run [44603384206](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44603384206) failed at the expected `CnotStages.initialize` callback (1 failed, 223 passed, 2 skipped; artifact 14570450432). An intermediate run [44604046633](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44604046633) stopped at rustfmt before tests (artifact 14569182918). Branch run [44604475999](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44604475999) passed CPU/PyO3 validation (224 passed, 2 skipped; artifact 14570037387).
- A review found and removed an unnecessary narrowing of the owner scalar; Rust now preserves Python's `owner == rank` comparison. Corrected branch run [44604945533](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44604945533) passed Rust checks, PyO3 build, 224 tests (2 skipped), and the NCCL probe syntax check. Artifact [14570022898](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44604945533/artifacts/14570022898), SHA-256 `7192da508d6bfa3f05dbcf2b4ba05fd2f85c5a4d07ebbf1c7badd22297f30a84`.
- - MAIN verification after fast-forward: hosted run [44606108341](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44606108341) passed Rust checks, PyO3 build, 224 CPU/PyO3 tests (2 skipped), and NCCL probe syntax checking. Artifact [14569653026](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44606108341/artifacts/14569653026), SHA-256 `b4ad12f9dda7dab2583a1a2b819d8a1447413a3b0e27d1d2f95ad3915f021ee5`.
This is hosted CPU/PyO3 verification only. Physical CUDA/NCCL multi-GPU execution remains unverified. Evidence: `test_results/refactor_20261119/cnot-rust-initialize/GITHUB_STATUS.md`.

## 2026-11-19 - Move CNOT Stream 5 startup dispatch into Rust

- `run_cnot_native_search` now calls the existing Rust Stream 5 setup helper directly. The helper keeps the same Rust-versus-Python transport choice, rank-wide capability vote, Python transport objects and `world.transport` assignment. The Python `World.start_transport` method remains for compatibility with older extension paths.
- RED Action [44580579687](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44580579687) failed at the removed `World.start_transport` call (218 passed, 2 skipped). GREEN Action [44581088466](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44581088466) passed Rust checks, PyO3 build, 219 Python tests (2 skipped), and NCCL probe syntax. Artifact [14559050896](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44581088466/artifacts/14559050896), SHA-256 `c74344cd0f026fa31d2ae70264023d438c86649f551a2e7e174073f5814d7ba8`.
- This is hosted CPU/compile verification. No CUDA execution or multi-GPU NCCL runtime was performed. Details are in `test_results/refactor_20261119/cnot-rust-stream5-dispatch/GITHUB_STATUS.md`.


## 2026-11-19 - Move CNOT admission collective dispatch into Rust

- `run_cnot_native_search` now calls the existing Rust helpers directly for root/config/scorer consensus and the all-rank native-stage support vote. Python `World` still supplies the same process-group collective primitives and the Stream 5 startup adapter. CNOT search ordering, fallback stage selection, transport, scoring, and state equality are unchanged.
- The RED run [44576411236](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44576411236) failed at the expected legacy `World.require_consensus` call (218 passed, 2 skipped). The final CPU run [44577927456](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44577927456) passed Rust checks, PyO3 build, 219 Python tests (2 skipped), and NCCL probe syntax. Artifact [14556663231](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44577927456/artifacts/14556663231), SHA-256 `e78d64e5ed298260ac113b20f0cb6b3b419ff56a0b1cbdbbc6a20f76b2ca5b6d`.
- Hosted CPU/compile verification only. It did not run CUDA or a physical multi-GPU NCCL collective. Details and the intermediate test-double correction are recorded in `test_results/refactor_20261119/cnot-rust-admission-vote/GITHUB_STATUS.md`.


## 2026-11-19 - Rust owns CNOT checkpoint scorer metadata validation

- The native-sys `TqpScorer` now parses and validates the JSON contract for 102 registered feature inputs and stores `max_packet_rows`; Rust computes the effective inference batch and closes the native handle if metadata validation fails. PyO3 still materializes the same public Python metadata dictionary. The scorer ABI, chunk boundaries, checkpoint model, output ordering, and numerical policy are unchanged.
- RED Action [44569025127](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44569025127) failed because the new tests referenced the not-yet-implemented Rust metadata owner. GREEN Action [44570920822](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44570920822) passed Rust formatting/checks, 8 native scorer tests, PyO3 build, 219 Python tests (2 skipped), and NCCL probe syntax. Artifact [14553567932](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44570920822/artifacts/14553567932), SHA-256 `d24e4df850867c6f291ff6bb736eac3a4b2fc3846ed6f50c1ec1b8c11737bc54`.
- These are hosted CPU/compile checks. No physical CUDA or multi-GPU NCCL scorer run was performed. The verification note is `test_results/refactor_20261119/cnot-scorer-metadata-contract/GITHUB_STATUS.md`.

## 2026-11-19 - Preserve Python inference selection while moving its owner to Rust

- Added a `beam-core` implementation of the existing profile selection behavior, including the CPython-compatible deterministic bootstrap and rejection messages. The optional PyO3 adapter is used for ordinary `Measurement` records; older extensions and unusual Python values retain the Python fallback.
- GitHub Actions [44554484032](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44554484032) passed formatting, scoped Rust checks, the PyO3 build and 217 CPU tests (2 skipped). NCCL probe syntax passed, but no physical multi-GPU collective ran. Full evidence: `test_results/refactor_20261119/rust-inference-selection/GITHUB_STATUS.md`.
- This moves control/statistics ownership without changing the search algorithm, scores, precision, GPU stages or multi-GPU transport. Physical CUDA/NCCL acceptance remains unverified.

## 2026-11-19 - Align the Rust identity builder test with session reuse

- The owner-reuse test now expects one identity build for the initial configuration and one after the model artifact changes. It checks that the second Rust-built identity carries the new model hash. Reusing an unchanged session does not rebuild this identity.
- This corrects test expectations only; production search and multi-GPU behavior are unchanged. Hosted Python-owner run [44535162475](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44535162475) passed (83 passed, 3 skipped). CNOT CPU run [44535162234](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44535162234) passed Rust CNOT checks, PyO3 build, 180 tests (2 skipped), and NCCL probe syntax. Artifact digests and physical-GPU limits are recorded in `test_results/refactor_20261119/rust-session-identity-builder/GITHUB_STATUS.md`.

## 2026-11-19 - Rust owns native outcome-file persistence

- Rust resident sessions now keep file ownership through result finalization. Python still returns the same NativeOutcome, but skips its intermediate native-outcome.json write only when the active Rust worker has a finalizable outcome. Rust atomically writes callback and persistent-session metadata. Calls without an active capable Rust owner, Python sessions, and unsupported extensions preserve the fallback JSON bytes, including the trailing newline.
- Search stages, CNOT semantics and rank ordering are unchanged. Hosted evidence and its limits are recorded in test_results/refactor_20261119/rust-outcome-file-owner/GITHUB_STATUS.md.

## 2026-11-19 - Rust owns prepared rank-group invalidation

- `NativeSession` delegates preparation transitions to Rust. An equal typed
  identity retains the resident rank group; a changed identity closes old
  workers and clears session metadata before Python's next preparation callback.
  PyO3 releases the GIL while Rust closes workers. Older extensions retain the
  `preparation_matches` and Python fallback.
- CNOT CPU workflow [44481115716](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44481115716)
  passed Rust formatting/tests, the PyO3 build, 174 Python tests (2 skipped),
  and NCCL probe syntax. The execution log is in artifact
  [14507489644](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44481115716/artifacts/14507489644);
  summary and test counts are in
  `test_results/refactor_20261119/rust-preparation-transition/GITHUB_STATUS.md`.
- This changes only prepared-session ownership. Search stages, multi-GPU ordering,
  CUDA execution, and result formats are unchanged. Physical CUDA/NCCL multi-GPU
  runtime remains unverified.

## 2026-11-19 - Restore the full multi-GPU architecture and session test contract

- Recovered the complete architecture baseline from commit `0f4729b8a7d65fc1b4c41dfbccd83361e82044c0` after discovering that commit `71fc8150f2aaa4935f9627ba097cb1921ec48cd4` had removed the CUDA memory, dispatcher, Stream 1–5, and multi-rank sections. Later Rust/CNOT ownership updates are retained in a dated addendum after the restored baseline.
- Restored the original `beam-runtime/tests/search_session.rs` suite and moved the preparation-identity regression into its own test file. The earlier Action `44436290093` ran against the replacement one-test file and does not verify the original session suite; the corrective hosted run is recorded in `test_results/refactor_20261119/rust-session-identity/GITHUB_STATUS.md`.
- Corrective hosted run [44442228498](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44442228498) passed the Rust checks, restored `search_session.rs` suite (2 passed), separate preparation-identity test (1 passed), PyO3 build, Python compatibility suite (174 passed, 2 skipped), and NCCL probe syntax check. Artifact [14490574306](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44442228498/artifacts/14490574306) has SHA-256 `103d628d22a539f08646ce37e58d89ce9d45ecb4c0d1d1e7598bfbaf8098ef68`.
- Restored the full changelog after its later truncation. Physical CUDA/NCCL multi-GPU runtime remains unverified.

## 2026-11-19 - Rust owns the prepared session reuse key

- `beam-runtime::SearchSession` now stores and compares the prepared-session
  identity key through optional PyO3 methods. `NativeSession` delegates the
  Rust-engine reuse decision to that owner; old extensions retain the Python
  comparison fallback, and `engine="python"` is unchanged. The actual typed
  `SessionIdentity` validation, rank launch, CUDA/NCCL stages, and result schema
  are unchanged.
- Test-first Action `44434253940` failed on the three new expectations
  (76 passed, 3 failed, 1 skipped). After formatting correction, Action
  `44436290093` passed Rust checks, the PyO3 build, 174 Python tests (2 skipped),
  and the NCCL probe syntax check. Artifact `14489551078` is recorded in
  `test_results/refactor_20261119/rust-session-identity/GITHUB_STATUS.md`.
- Depth-control CPU validation `44436290039` passed. No physical GPU/NCCL
  search was run. Existing Actions secret `VAST_API_KEY` passed the read-only
  status check in run `44427129602`; that check reported the existing instance
  stopped, and no instance state or budget changed.

## 2026-11-19 - CNOT compatibility root check uses the Rust matrix owner

- Exposed the existing packed-matrix identity check on the PyO3
  `CnotSearchInput` and `CnotMatrix` types. The compatibility CNOT stage now
  uses that Rust-owned matrix when available and retains `torch.equal` for
  older extensions and Python-only input. Search order, CUDA operators,
  scoring, collectives, and state storage are unchanged.
- Test-first run 44429473842 failed only on the two new root-check cases
  (165 passed, 2 skipped), as expected. Green branch run 44430494875 and
  published-main run 44431181310 each passed Rust checks, PyO3 build,
  167 Python tests (2 skipped), and NCCL probe syntax. Their artifacts are
  recorded in `test_results/refactor_20261119/cnot-compat-root-goal/GITHUB_STATUS.md`.
- Physical CUDA/NCCL and multi-GPU CNOT execution remain unverified; the
  existing Vast instance is stopped. No GPU instance state changed.

## 2026-11-18 - Rust controls permutation depth continuation

- Preserved the original ABI-1 depth callback layouts and two checkpoints, and
  added an optional ABI-2 table for solve-bucket stop decisions. Rust owns the
  calibration-limit, frontier-empty, debug-path, and ABI-2 solve-bucket policy.
  C++ still owns the CUDA stages, collective calls, bucket scan, solution paths,
  and depth finalization. Callbacks carry scalar state only and never run per
  candidate or kernel.
- `beam-runner` installs the callback for one-shot and persistent Rust paths.
  Older native libraries and direct C++ launches keep the original C++ rules.
- The hosted CPU workflow is
  `.github/workflows/permutation-depth-control-cpu.yml`. CUDA runner compilation,
  NCCL runtime, and physical multi-GPU behavior remain separate acceptance work.
- Commit `697c1d1437d2e4f39cf3cc9c1115280c9e865810` keeps ABI-1 descriptor
  layouts and the Rust `DepthControlSink` contract intact, while adding separate
  ABI-2 symbols and layouts for solve-bucket state. GitHub Actions runs
  `44384857755` (depth-control) and `44384857776` (CNOT CPU compatibility) passed;
  their artifacts are recorded in
  `test_results/refactor_20261118/permutation-depth-control/GITHUB_STATUS.md`.

## 2026-11-18 - CNOT stage error coordination through Rust

- Multi-rank `World.checked` now delegates callback execution and the existing
  rank-wide error vote to the PyO3 `run_cnot_native_checked` boundary when the
  symbol is available. The operation stays on the calling thread and keeps the
  same `world.sum` then failure-only `all_gather_object` sequence.
- Older extensions retain the Python fallback, and one-rank execution is
  unchanged. Added tests for Rust delegation and the old-extension fallback.
  Hosted run 44352267137 passed Rust formatting/checks, PyO3 build, the CNOT CPU
  suite and NCCL probe syntax; artifact 14448882195 has SHA-256
  `4c3f4587e51f2a73e20c6f522e6d49faf9e223923f2fc8b3441b2291757a0ef8`. The
  associated Vast preflight run 44350398801 confirmed the Actions secret is
  available but skipped GPU execution because the existing instance was stopped.
- The follow-up row-gather contract first failed as expected in run
  44354152193 (2 failed, 135 passed, 2 skipped): the extension had no
  `run_cnot_native_gather_rows` entry point and `World.gather_rows` did not
  delegate. Its full log remains in the Actions artifact. Rust row-gather
  implementation passed hosted run 44355550826: Rust formatting/checks,
  PyO3 build, 137 Python tests (2 skipped), and the NCCL probe syntax check.
  Artifact 14451745790 has SHA-256
  `6a624e57716f68e3fa315232418dccce96cd489e9f31d2337af2ed5c9c0783ac`. Vast
  preflight 44355550941 confirmed the repository secret is available but
  skipped NCCL execution because the existing instance was stopped; no instance
  state change was attempted. No local tests or builds were run. Physical
  CUDA/NCCL behavior and full migration remain unverified.

## 2026-11-18 - CNOT search control validation in beam-core

- A compatible PyO3 extension now validates the CNOT wire count, global width,
  shortlist and score mode through `beam-core`. The signed lexical-key width
  bound and public error messages are preserved.
- Python keeps exact Python-type and depth checks, the `validation="python"`
  route, and the fallback for older extensions. CNOT search stages, tensor and
  scorer calls, collectives and rank topology are unchanged. Hosted CPU
  validation is pending; this does not claim CUDA/NCCL runtime acceptance.

## 2026-11-18 - GraphContract construction in beam-core

- The CayleyPy adapter continues to extract list/tuple, NumPy and Torch values
  and reject type coercion. A compatible PyO3 extension now sends normalized
  graph values to `beam-core`, which validates state labels, the start-state
  multiset, generator permutations and ordered generator count, then returns
  the contract and the same canonical graph SHA-256 used by CayleyPy. The
  persistent Rust session now calls this shared hash implementation.
- Python-only and older-extension fallback remains. Search stages, model
  preparation, CNOT execution, CUDA/NCCL, candidate order and multi-GPU
  topology are unchanged. Hosted run 44335585382 passed Rust formatting,
  3 `beam-core` CNOT tests, 9 `beam-core` search tests, 4 `beam-runtime`
  CNOT tests, 4 `beam-native-sys` scorer tests, the PyO3 check/build, 129
  Python tests (2 skipped), and the NCCL probe syntax check. Artifact
  14441861160 SHA-256: `b50a75854f46fc34363189ddc93ccd43baba81156f2cc0ca87dc1ba7f7786cac`.
- Initial run 44333600886 passed Rust checks and PyO3 build but test collection
  found that pinned `cayleypy==0.1.0` does not expose the private
  `cayleypy.models.checkpoint` module. The replacement fixtures use the public
  graph hash schema and retain the existing Rust session identity test. This
  is hosted CPU/control-flow evidence only; CUDA/NCCL runtime and physical
  multi-GPU acceptance remain unverified.

## 2026-11-18 - Rust validation for normalized public search controls

- After Python binds arguments against the installed CayleyPy signature, it now passes normalized search controls to `beam-core`: beam width (including `"max"`), maximum depth, beam mode, taboo history, destination/BFS compatibility, and boolean flags.
- `beam-core` distinguishes invalid values from unsupported public modes; the PyO3 adapter returns those outcomes to Python without changing their public exception classes or messages. Existing Python validation remains for old extensions and custom objects whose comparison semantics cannot be represented by the adapter.
- This changes validation ownership only. Search decisions, candidate order, C++/CUDA workers, and multi-GPU topology do not change. Hosted CPU validation passed in Actions run 44249581929, including 125 Python tests, Rust checks, and the PyO3 build. CUDA, NCCL, and physical multi-GPU runtime remain unverified.

## 2026-11-18 - Rust владелец публичных native-сессий по умолчанию

- Публичный native dispatch и `NativeSession()` теперь выбирают Rust-владельца
  ранговых работников, когда доступны совместимый PyO3 API и `beam-runner`.
  Старые установки автоматически сохраняют прежний Python-владелец; доступны
  явные `engine="python"` и `engine="rust"`.
- Выбор проверяет API и исполняемый worker до подготовки модели и CUDA-сборки.
  `backend="torch"` остаётся без изменений. Подготовка модели, калибровка и
  вычислительный C++/CUDA runner ещё не перенесены этим шагом.
- RED run 44181078596 подтвердил два отсутствующих поведения (2 failed, 2
  passed). GREEN run 44182404997 прошёл 78 тестов за 5.19 секунд и загрузил
  artifact 14372272952 (SHA-256 `40342afa5079cfc7121667be570a7d818c02f2f2c7fae52f9f1af2b34790f73f`).
  Это hosted CPU evidence: feature selection замокан; аппаратный worker,
  CUDA/NCCL и реальная multi-GPU работа этим прогоном не проверялись.

## 2026-11-19 - Смена identity через существующий Rust session owner

- `NativeSession` оставляет существующий PyO3 `SearchSession` при смене модели,
  геометрии, опций или runtime-профиля и передаёт новые identity/launch в Rust.
  Rust сравнивает identity и сам закрывает несовместимую rank-группу перед
  запуском нового задания. Если подготовленный вызов завершается до launch,
  Python освобождает старую сессию, чтобы не удерживать устаревшие workers.
- Identity включает Python-only поля геометрии, размеров model artifact,
  binding scope и `repr(options)`. Проверка request count принимает сброс к 1
  только вместе с `persistent_session.reused=false`. Python fallback, формат
  результата, C++/CUDA stages и multi-GPU topology не менялись; оптимизации
  поиска не добавлялись.
- RED run 44414135860: ожидаемо упали две параметризации теста повторного
  создания owner (76 passed, 2 failed). GREEN run 44415524622: 78 passed за
  5.48 секунды. Артефакты и SHA-256 записаны в
  `test_results/refactor_20261119/rust-session-reuse/GITHUB_STATUS.md`.
  Это hosted CPU проверка Python owner; реальный PyO3 runtime, CUDA/NCCL и
  физическая multi-GPU работа этим запуском не проверялись.
- Изменения опубликованы fast-forward в `main` на `7e07a5a5`. Main Actions run
  44417405012 также прошёл: 78 passed за 5.59 секунды; артефакт и SHA-256
  указаны в status note.

## 2026-11-18 - CNOT identity-root control in Rust

- Перенесено решение для уже целевого начального CNOT-состояния из общего
  Python dispatch в Rust CNOT controller. Проверка выполняется до
  инициализации ранговых стадий и возвращает прежний `solved` с пустым путем,
  без слоев и маршрутов; Python reference route сохранён.
- Добавлены CPU Rust/Python проверки и сохранение полных логов CNOT workflow в
  `test_results/refactor_20261118/cnot-root-goal/`. RED run 44184512164
  подтвердил пропущенный Rust-вызов (1 failed, 19 passed). GREEN run
  44186094978 проверил 3 `beam-core` CNOT tests, 4 Rust controller tests,
  `cargo check -p beam-python` и 20 Python tests. Artifact 14375525292 имеет
  SHA-256 `a9a783ccf03cf7caa6080fb3a7d4083c4671a2066dbaf80e2bd7bf220b295299`.
- Промежуточный run 44185419853 остановился до тестов из-за workspace-wide
  rustfmt-разниц, включая два неизменённых файла; проверка ограничена тремя
  изменёнными Rust-файлами. Это hosted CPU evidence, не CUDA/NCCL acceptance.

## 2026-11-17 - Vast key validation and status publication

- Confirmed the existing `VAST_API_KEY` Actions secret is populated and accepted by the read-only status request for instance 61080396. At 18:03 UTC it reported `exited`, intended `stopped`, with 2 x RTX 3060; no instance was started.
- Added the missing `status` step id required by the conditional report publisher. Run 44143759674 had skipped publication; follow-up run 44148276778 published the sanitized JSON. At 18:28 UTC it reported `exited`, intended `stopped`, with 2 x RTX 3060; no instance was started.

## 2026-11-17 - Default Rust CNOT input validation

- `search_cnot` and the CNOT CLI now prefer the Rust packed-matrix validator when installed. Explicit `python` remains available; explicit `rust` fails closed; `auto` falls back to Python when the extension capability is absent.
- Added a GitHub-hosted CPU verification workflow. Run 44150899645 produced the expected RED: the two new cases failed for missing `auto` behavior, and the other 17 passed. Run 44153448093 passed three `beam-core` CNOT tests, three Rust search-controller tests, the PyO3 compile check, and all 19 Python CNOT tests. It uploaded artifact 14360561194. This is CPU-only evidence; CUDA/NCCL multi-GPU validation remains pending.

## 2026-11-17 - Rust CNOT search descriptor validation and recovery test

- Published the opt-in Rust CNOT search route for root ownership, batched child
  generation and generated-child ownership through the existing borrowed CUDA
  ABI. The default and CPU paths, scorer, numerical policy, full-matrix
  deduplication, NCCL schedule and history behavior remain unchanged. This is
  an incremental migration stage, not full CNOT or beam-search completion.
- Replaced the borrowed expansion bridge's asynchronous device assertion with
  descriptor range checks and host reads of one boolean per descriptor tensor.
  An invalid descriptor returns before the native kernel and reaches the
  existing rank-wide Rust stage vote.
- The previous two-rank descriptor recovery command exited nonzero. Source
  review found a likely test-target error: the test injected on the root hash
  owner, while the initial expansion frontier is on rank 0; the fixture's
  two-rank hash owner is rank 1. The test now injects on rank 0. This diagnosis
  is source-derived because the full remote log remains on Vast; the corrected
  rerun and same-group recovery are pending.
- The read-only Vast status workflow now has a path-scoped main push trigger,
  sanitized job output and a configured small status note in `test_results/`.
  Its latest Actions result has not been read back, so current instance state
  remains unconfirmed; no remote compute was started by this change.

## 2026-11-17 - Opt-in CNOT search device selection bridge

- Connected the borrowed full-matrix first-unseen ABI to the Rust CNOT search
  through the explicit `MGBS_CNOT_OWNED_LIBRARY` path. Each rank retains its
  own context and Torch tensor storage; existing NCCL communication, scorer,
  default path and CPU behavior remain unchanged. The bridge reads the device
  count on the host to construct the selected result view.
- One- and two-RTX-3060 comparisons covered repeated calls, mixed rank paths,
  a coordinated one-rank load error and the existing checkpoint scorer. A
  CUDA invalid-handle API diagnostic also reproduced in the old Torch owner
  operation; the source-pinned memory rerun disabled API-error reporting.
  Details and limits: `test_results/refactor_20261117/STAGE_CNOT_SEARCH_DEVICE_BRIDGE.md`.

## 2026-11-17 - Borrowed CNOT full-matrix distinct stage

- Added an optional Rust/CUDA first-unseen ABI with caller-owned device inputs,
  outputs and explicitly sized CUB workspace. It preserves the existing full
  matrix comparison and first occurrence order; empty batches set the device
  count to zero. Existing libraries remain loadable. This standalone operation
  does not itself switch the public CNOT search.
- Two RTX 3060s passed the full-matrix oracle, Rust ownership, memory and race
  checks; the Rust workspace and SM75/80/90/120 cross-compilation passed.
  The initial test-only pinned host cache diagnostic and verified rerun are
  retained in `test_results/refactor_20261117/STAGE_CNOT_BORROWED_UNSEEN.md`.

## 2026-11-17 - Borrowed CNOT child generation

- Added an optional device-only CNOT context and borrowed CUDA expansion using
  the existing kernel and int64/uint16 move formats. Caller-owned buffers and
  stream remain subject to explicit bounds, lifetime, descriptor and ordering
  contracts; older libraries and the full host-buffer context remain supported.
  This is standalone and does not switch public search or alter NCCL.
- Two RTX 3060s passed CPU-oracle tests, Rust ownership checks and CUDA
  memcheck/racecheck. One/two-GPU search and native checkpoint scorer checks,
  the Rust workspace and 835 Python tests passed; 10 Python tests were skipped.
  See `test_results/refactor_20261117/STAGE_CNOT_BORROWED_EXPANSION.md` for
  source-bound logs, the corrected compatibility fixture and hardware limits.

## 2026-11-17 - Borrowed CUDA views for standalone CNOT ownership

- Added an optional versioned device-view extension for the existing CNOT
  owner kernel and its unsafe Rust boundary. It accepts caller-owned CUDA
  allocations and a stream, validates declared extents, devices and overlap,
  and synchronizes the stream before returning. The base host ABI and old
  libraries remain supported. Public search and NCCL scheduling are unchanged.
- Source-pinned one/two RTX 3060 checks, Rust tests, CUDA memcheck/racecheck,
  833 Python tests and SM75/80/90/120 cross-compilation passed. Twelve Python
  tests were skipped. The standalone extension is not yet connected to search.
  The test diagnostics, continuation, hardware limits and provider release are
  recorded in `test_results/refactor_20261117/STAGE_CNOT_BORROWED_DEVICE.md`.

## 2026-11-17 - Standalone Rust-owned CNOT CUDA domain

- Added an optional versioned C ABI and Rust owner for the existing CNOT
  expansion, owner hashing and full-matrix distinct kernels. The context owns
  one device, stream, fixed-capacity buffers and queried CUB scratch. Its
  current host-slice interface is standalone and does not switch search.
- Two RTX 3060s passed the domain CPU-oracle and Rust ownership checks, plus
  CUDA memcheck and racecheck. Existing one/two-GPU CNOT searches, the Rust
  workspace and the Python suite passed; SM75/80/90/120 cross-compilation passed.
  See `test_results/refactor_20261117/STAGE_CNOT_OWNED_DOMAIN.md` for exact
  source binding, skips, release receipt and unverified hardware.

## 2026-11-17 - Rust-owned CNOT record exchange orchestration

- Moved the CNOT stage backend's stable record grouping and exchange dispatch
  into Rust. The same C++ ATen or Torch packet operations, Stream 5 transport,
  CPU/Gloo collectives, empty splits and rank ordering remain in use.
- Source-pinned Rust, 95 focused CPU/Gloo/CUDA-controller, 62 CUDA and 835
  Python integration tests passed (10 skipped). One/two RTX 3060 searches,
  empty and mixed-operator NCCL exchanges and checkpoint scorers matched
  their references. See `test_results/refactor_20261117/STAGE_CNOT_NATIVE_RECORD.md`
  for evidence and remaining hardware limits.

## 2026-11-17 - Rust-owned CNOT batch orchestration

- Moved score validation and the existing CPU/older-CUDA candidate expansion
  and distinct-preparation sequence into the Rust stage backend. Torch retains
  tensor storage, and C++ ATen/domain operators, rank collectives, scoring,
  ordering and numerical policy remain unchanged. The older Python stage
  adapter remains available for compatible older extensions.
- Source-pinned Rust, 91 focused CPU/Gloo/CUDA-controller, 62 CUDA and 831
  Python integration tests passed (10 skipped). One/two RTX 3060 CNOT searches,
  mixed old/new operators and checkpoint scorers matched their references.
  See `test_results/refactor_20261117/STAGE_CNOT_NATIVE_BATCH.md` for evidence
  and remaining hardware limits.

## 2026-11-17 - Rust-owned CNOT stage error coordination

- Moved the rank-wide error vote and failure-message gather for CNOT stage
  initialization, expansion and distinct-child evaluation into the Rust stage
  backend. The Python admission and legacy backend stay compatible; collective
  order and original exceptions are preserved.
- Source-pinned Rust, 80 focused CPU/Gloo/CUDA-controller, 62 CUDA and 820
  Python integration checks passed (10 skipped). NCCL error injection,
  one/two-RTX-3060 searches and checkpoint scorers matched the reference.
  Evidence and remaining limits:
  `test_results/refactor_20261117/STAGE_CNOT_NATIVE_CHECKED.md`.

## 2026-11-17 - Rust-owned CNOT goal coordination

- Transferred rank-local goal orchestration and the existing rank-wide
  winner/error exchange into the Rust stage backend. Torch retains tensor
  storage and the same CPU or C++ ATen operations; the old Python backend
  remains available for older extensions. No search or numerical policy changed.
- Source-pinned Rust, 72 focused CPU/Gloo/CUDA-controller, 62 CUDA and 812
  Python integration checks passed (10 skipped). One/two RTX 3060 searches,
  mixed-operator NCCL ranks and checkpoint scorers matched their references.
  See `test_results/refactor_20261117/STAGE_CNOT_NATIVE_GOAL.md` for complete
  evidence and remaining hardware limits.

## 2026-11-17 - Rust-owned CNOT global selection stages

- Moved the rank-local global cutoff keys, masks and batch-stage ownership to
  Rust, preserving the same ATen operators, radix schedule, stable ordering,
  collectives and older Python selection adapter. PyTorch continues to own
  tensor storage.
- Source-pinned Rust, 53 focused CPU/Gloo/CUDA-controller, 62 CUDA and 793
  Python integration checks passed (10 skipped). One/two RTX 3060 searches and
  checkpoint scorers matched their references, including mixed old/new CUDA
  operators across ranks. Filtered Compute Sanitizer results and the remaining
  Python boundaries are recorded in
  `test_results/refactor_20261117/STAGE_CNOT_NATIVE_SELECTION.md`.

## 2026-11-17 - Rust-owned CNOT rank-local stages

- Added a Rust stage backend that retains each rank's frontier, closed-set,
  candidate, metadata and history tensor handles while PyTorch owns their
  storage. All ranks agree on availability and jointly fall back to the old
  Python adapter if one rank has an older extension. Existing operators,
  numerical policies, record routing and NCCL exchange order are preserved.
- Source-pinned Rust, 45 CPU/Gloo, 62 CUDA and 785 Python integration checks
  passed (10 skipped). One/two RTX 3060 searches and checkpoint scorers matched
  their references. Filtered Compute Sanitizer memcheck reported zero errors;
  unfiltered NCCL API diagnostics remain a known limitation. See
  `test_results/refactor_20261117/STAGE_CNOT_NATIVE_STAGES.md`.

## 2026-11-16 - Rust CNOT Stream 5 record transport

- Moved the CNOT Stream 5 count and payload exchange sequence into the Rust
  controller, with checked borrowed CUDA tensors and the existing NCCL/CUDA
  operations. All ranks agree on the Rust transport or the compatible legacy
  fallback; CPU, single-rank and Python-reference paths remain available.
- Source-pinned checks and their hardware limits are recorded in
  `test_results/refactor_20261116/STAGE_CNOT_STREAM5_TRANSPORT.md`.

## 2026-11-16 - Device-bound CNOT Stream 5 owner

- Added a separate owned native ABI that remembers the communicator's CUDA
  device, selects it for close or abort, and restores the calling thread's
  current device. The existing Python transport's raw ABI is unchanged; the
  Rust owner remains experimental and thread-bound.
- Source-pinned Linux comparisons, including two-rank device-switch lifecycle
  checks, are recorded in `test_results/refactor_20261116/STAGE_CNOT_STREAM5_DEVICE.md`.

## 2026-11-16 - Rust-owned CNOT Stream 5 communicator lifecycle

- Added a thread-bound Rust owner for the versioned native CNOT Stream 5
  library and NCCL communicator, exposed for comparative tests through PyO3.
  The existing Python transport remains active and no GPU buffers cross the
  new interface. This first version required the same current device for
  initialization and close; the later device-bound owner removes that constraint.
- Source-pinned Linux checks passed on one/two RTX 3060 GPUs: Rust workspace,
  39 focused CPU/Gloo tests, 62 CUDA cases, three NCCL lifecycle rounds per
  rank, complete searches, checkpoint comparisons, and the full Python suite
  (779 passed, 10 skipped). See
  `test_results/refactor_20261116/STAGE_CNOT_STREAM5_RUST.md`.

## 2026-11-16 - C++ ATen CNOT selection preparation and ordering

- Moved CUDA key preparation, selected-record creation and final ordering for
  the Rust CNOT radix controller into C++ ATen. The same rank collectives,
  deterministic ordering, CPU path and older-library fallback are retained.
- Source-pinned Linux checks on one/two RTX 3060 GPUs passed: Rust workspace,
  34 focused CPU/Gloo tests, 62 CUDA cases, full Python suite (774 passed, 10
  skipped), complete searches, NCCL tests and checkpoint comparisons. See
  `test_results/refactor_20261116/STAGE_CNOT_SELECTION_CPP.md`.

## 2026-11-16 - C++ ATen CNOT exchange packet preparation

- Moved CUDA owner counting and stable record grouping for the Rust CNOT
  controller's multi-rank exchanges into C++ ATen. Stream 5 transport, the
  single-rank path and compatibility fallbacks remain unchanged.
- Source-pinned Linux checks on one/two RTX 3060 GPUs passed: Rust workspace,
  34 focused CPU/Gloo tests, 54 CUDA cases, full Python suite (766 passed, 10
  skipped), complete searches, NCCL tests and checkpoint comparisons. See
  `test_results/refactor_20261116/STAGE_CNOT_EXCHANGE_CPP.md`.

## 2026-11-16 - C++ ATen CNOT radix tensor stages

- Moved CUDA histogram calculation and mask updates for CNOT radix selection
  into C++ ATen without changing the cutoff algorithm or collectives. CPU and
  older external operator libraries retain their existing paths.
- Source-pinned Linux checks on one/two RTX 3060 GPUs passed: Rust workspace,
  34 focused CPU/Gloo tests, 49 CUDA cases, full Python suite (761 passed, 10
  skipped), complete searches, NCCL tests and checkpoint comparisons. See
  `test_results/refactor_20261116/STAGE_CNOT_RADIX_CPP.md`.

## 2026-11-16 - C++ ATen CNOT local goal

- Moved CUDA local-goal evaluation into C++ ATen for the Rust controller.
  Rank-local failures still pass through the existing coordinated goal exchange;
  CPU, Python reference and older external CUDA libraries retain their paths.
- Source-pinned Linux checks on one/two RTX 3060 GPUs passed: the Rust workspace,
  34 focused CPU/Gloo tests, 36 CUDA cases, full Python suite (748 passed, 10
  skipped), full searches, NCCL goal/radix tests and checkpoint comparisons.
  See `test_results/refactor_20261116/STAGE_CNOT_GOAL_CPP.md`.

## 2026-11-16 - C++ ATen CNOT distinct preparation

- Moved generation-order sorting and preparation of distinct CUDA CNOT records
  into C++ ATen. The existing full-matrix deduplication kernel, Python scorer,
  multi-rank exchange and compatibility fallback remain in use.
- Source-pinned Linux tests on one/two RTX 3060 GPUs passed: the Rust workspace,
  34 focused CPU/Gloo tests, 26 CUDA cases, full Python suite (738 passed, 10
  skipped), NCCL comparisons and checkpoint comparisons. See
  `test_results/refactor_20261116/STAGE_CNOT_DISTINCT.md`.

## 2026-11-16 - C++ ATen CNOT candidate expansion

- Moved the CUDA CNOT batch expansion tensor sequence into C++ ATen, retaining
  the existing CUDA children and owner kernels, stable action order, full-word
  state handling and physical routes. Arbitrary Python scorers still run as
  batch providers; CPU and old external operator paths remain available.
- The source-pinned one/two-RTX-3060 comparison and CUDA and full integration
  checks are recorded in `test_results/refactor_20261116/STAGE_CNOT_EXPAND.md`.

## 2026-11-16 - Rust control of the CNOT global cutoff

- Moved the score and lexical radix cutoff pass sequence and boundary decisions
  into the Rust CNOT controller. Python still performs the same batched Torch
  histograms and rank collectives; scoring, order, and full matrix equality are
  unchanged. Old extensions and old stage adapters retain their existing path.
- Source-pinned checks passed on Linux with two RTX 3060s: Rust workspace,
  34 focused CPU/Gloo tests, eight CUDA cases, the complete Python suite (720
  passed, 10 skipped), two-rank NCCL radix cases and one/two-GPU search and
  checkpoint comparisons. See
  `test_results/refactor_20261116/STAGE_CNOT_RUST_SELECTION.md`.

## 2026-11-16 - CNOT Rust controller and native frontier packing

- Added a Rust rank-local CNOT search controller and history reconstruction.
  A compatible PyO3 module is selected automatically; explicit Python and
  Rust engine modes remain available. Python tensor/scorer/collective stages
  remain transitional and retain their existing order and error coordination.
- Moved CUDA winner-packet creation, received-record sorting and frontier
  views to C++ ATen operations. Older external operator libraries retain the
  previous tensor path. The packed-matrix equality, scorer, precision and
  multi-rank transport are unchanged.
- Source-pinned checks passed: full Rust workspace; 31 focused CNOT CPU/Gloo
  tests; eight CUDA frontier-packet cases; the full Python suite (717 passed,
  10 skipped); and exact Python/Rust one- and two-RTX-3060 comparisons with
  synthetic scoring and the existing checkpoint provider. Checkpoint qhat
  replayed `[5, 7, 0]`; proposal's feature coverage rejection matched on both
  controllers. See `test_results/refactor_20261116/STAGE_CNOT_RUST_CONTROLLER.md`.

## 2026-11-16 - Rust supervisor for externally scheduled ranks

- Added the opt-in Rust torchrun/Slurm rank supervisor and a thin PyO3 adapter.
  Rank validation, child launch, timeout, peer-failure handling and process
  cleanup now run in Rust for that mode. The Python supervisor remains the
  default; the native search, CUDA kernels and NCCL protocol are unchanged.
- The complete Rust workspace passed with two pre-existing ignored tests;
  37 focused process tests and the full Python suite (698 passed, 12 skipped)
  passed on Linux. Both supervisors passed a two-GPU NCCL exchange and the same
  real production-runner search on two RTX 3060s, with the same replayed path
  and exact global beam 64. Physical multi-node and real Slurm runs remain
  unverified. See `test_results/refactor_20261116/STAGE_RUST_CLUSTER_SUPERVISOR.md`.

## 2026-11-15 - Collective CNOT goal errors

- A local goal-evaluation exception is now carried in the existing goal
  exchange; every rank reports its source before leaving the search. The
  successful path does not add a collective or a host read. Single-rank
  exceptions retain their original type.
- Baseline RED and fixed GREEN tests cover each Gloo rank, simultaneous
  failures and group reuse; the full Linux suite reported 675 passed and 17
  skipped. One/two-RTX-3060 searches matched the original implementation;
  fault injection on each NCCL rank was followed by a successful replayed
  search. A broken device/communicator is outside this guarantee. See
  `test_results/refactor_20261115/STAGE_CNOT_GOAL_COLLECTIVE.md`.

## 2026-11-15 - Experimental CNOT checkpoint packet ABI

- Added an opt-in packet ABI that submits multiple existing checkpoint scorer
  sub-packets through one Python-to-native call. It checks declared buffer
  lengths and offset arithmetic before enqueueing; actual CUDA allocation,
  non-overlap and lifetime remain caller obligations. The existing ABI, model
  computations, coverage and finite checks remain the default.
- On both RTX 3060 devices the real pinned GL(4,2) checkpoint passed Qhat,
  proposal and V comparisons, empty/partial/multiple packet and short-buffer
  tests. Search results and independently replayed paths matched on one and two
  GPUs. The upstream canonical-output comparison was skipped in this run.
- Five paired complete searches gave median paired wall-time changes of -0.25%
  on one GPU and +0.03% on two; the tiny fixture is not a saturated benchmark.
  The option stays off. See `test_results/refactor_20261115/STAGE_CNOT_NATIVE_BATCH.md`.

## 2026-11-15 - Collective CNOT admission errors

- CNOT input validation, scorer metadata reads and configuration fingerprinting
  now share one rank-coordinated admission boundary. A metadata property or JSON
  encoding failure on one rank is reported to all ranks before digest exchange;
  the existing successful fingerprint format and collective count are unchanged.
- The unfixed source reproduced rank-local failures on two Gloo processes. The
  corrected source passed 674 Python tests (11 skipped), and one/two-rank CPU
  and RTX 3060 checks with the Python and Rust validators. Both ranks received
  each injected metadata error and successfully reused the process group.
  Twelve returned paths were independently replayed. The synthetic fixtures do
  not validate a trained checkpoint or complete the CNOT runtime migration; see
  `test_results/refactor_20261115/STAGE_CNOT_COLLECTIVE_ADMISSION.md`.

## 2026-11-15 Stream 4 conditional-graph investigation
- Extended GPU oracle fixtures for empty and singleton clean/dirty runs, valid
  all-ones hashes, duplicates across runs, and a singleton final union. The
  unchanged published index and dirty implementations passed direct tests on
  two RTX 3060s at capacities 1, 2, 3, 4 and 257, with scoped memcheck and
  ordinary-graph union synccheck passes.
- Bounded synccheck failures were reproduced. Skipping a one-item radix sort
  moved the reported error to subsequent kernels, without establishing a fix;
  that candidate was discarded and retained only as evidence. Minimal chained,
  prefixed and partially active-kernel variants did not reproduce the error.
  Bounded graph racecheck and synccheck remain open. See
  `test_results/refactor_20261115/STAGE_STREAM4_SYNCCHECK.md`.

## 2026-11-15 experimental borrowed-byte MLP validation
- Added a separately opt-in Rust/PyO3 finite-value validator for explicit MLP
  snapshots. It borrows immutable little-endian FP16/BF16 chunks and avoids
  the Python bytearray copy and tensor conversion; default validation remains
  unchanged. Incompatible opt-in configurations and extensions fail closed.
- The release Rust workspace and PyO3 build passed; 670 Python tests passed,
  11 skipped. All 131,072 FP16/BF16 encodings agreed with PyTorch. Public
  one/two-RTX-3060 tests covered 20 snapshot cases, with 16 paths independently
  replayed. Five paired two-GPU full calls measured median −2.01%, below the
  promotion threshold. No GPU-step or peak-memory claim follows.
- A separate minimal CUDA graph reproduced two cross-kernel racecheck reports
  in the conditional graph while its ordinary graph, memcheck and synccheck
  passed. The larger conditional racecheck run exited 11. The original Stream 4
  reports and bounded synccheck issue remain unresolved; the minimal example
  cannot certify application safety. See
  `test_results/refactor_20261115/STAGE_RUST_MLP_FINITE.md`.

## 2026-11-15 Rust terminal result transfer
- The opt-in Rust public session now hands its validated terminal result to
  Python once for the matching job rather than parsing the native log again.
  Stale sequences/results fail closed, old PyO3 modules are rejected, and
  per-job terminal metadata no longer inherits absent fields from the first run.
- Rust workspace tests and the Linux integration suite passed (658 passed,
  11 skipped). CUTLASS/LibTorch one/two-RTX-3060 tests covered 48 Python/Rust
  public scenarios; the source-bound verifier replayed 96 path entries. Five
  matched full-call pairs had a median paired wall change of -1.24%; variance
  and the 0.742 ms median extra parser time do not establish a GPU speedup.
  See `test_results/refactor_20261115/STAGE_RUST_TERMINAL_TRANSFER.md`.

## 2026-11-15 CNOT input snapshot ownership
- The CNOT adapter now validates and searches the same owned, contiguous
  device snapshot. A caller mutation after validation previously could change
  the searched matrix and even produce an incorrect empty solution. The one
  extra same-device copy per search is at most 8192 bytes; callers still need
  to synchronize writes during capture. The Python search loop is unchanged.
- A regression reproduced the old failure, then passed for both validators and
  contiguous/strided inputs. The Linux suite reported 651 passed, 11 skipped;
  public CPU/Gloo and RTX 3060/NCCL tests used the real PyO3 validator on one
  and two ranks. The source-bound receipt independently replayed 12 paths and
  checked 24 rank-local snapshot cases. See
  `test_results/refactor_20261115/STAGE_CNOT_INPUT_SNAPSHOT.md`.

## 2026-11-15 experimental Stream 4 dirty-run merge
- Added an off-by-default CMake path that compacts a sorted clean prefix, sorts
  only dirty Hash128/index pairs, merges both runs and applies the existing
  deterministic reduction. It reuses existing arenas, queries actual CUB
  scratch and preserves the final A/B union. The normal public build and
  specialized inference are unchanged.
- On two RTX 3060s, CPU-oracle GPU fixtures and compatible three-call resident
  sessions passed. Two matched five-pair full-step series measured median
  +1.91% and +1.56%; sampled peak memory was 2 MiB higher per GPU in the second
  series. This does not meet the default-promotion gate. Unfiltered racecheck
  and synccheck reported unresolved bounded-graph issues also reproduced in
  the previous index path. Compile-only SM75/80/90/120 checks do not establish
  runtime acceptance. Evidence and scoped cost are recorded in
  `test_results/refactor_20261114/STAGE_STREAM4_DIRTY_MERGE.md`.

## 2026-11-15 optional Rust CNOT input and path validation
- Added an owned Rust packed CNOT input contract with GF(2) invertibility,
  full-word construction replay, zero-tail validation and control-major action
  IDs. The opt-in Python/CLI adapter fails closed on missing Rust capability and
  requires the same validator on every rank. The Python search loop and trained
  checkpoint scorer are unchanged.
- Rust workspace tests (two ABI fixtures ignored) and release PyO3 build passed,
  along with 643 Python tests (11 skipped). Public synthetic CNOT searches agreed across validators on one
  and two CPU/Gloo and RTX 3060/NCCL ranks; twelve solved rank paths were
  independently replayed, including a 144-wire cross-rank two-gate path. Both
  distributed failure cases passed. No performance, checkpoint-quality or
  other-hardware claim is made. See
  `test_results/refactor_20261114/STAGE_CNOT_RUST_VALIDATION.md`.

## 2026-11-15 native-loaded permutation graph binding
- A Rust resident session now supplies its validated graph hash through an
  optional ABI-1 extension. Each native rank hashes its actual loaded center
  and ordered generators before model/GPU setup and rejects mismatches; the
  worker capability probe and missing-extension check fail closed. The normal
  legacy protocol response remains compatible. Bound sessions reject a changed
  target and the separate repair mode. CNOT and inference kernels are unchanged.
- Linux Rust, explicit FFI, ASan/UBSan and CTest checks passed; Python reported
  643 passed and 9 skipped. Four bounded public one/two-RTX-3060 searches had
  replayed paths; five mismatched-graph/target/mode cases, including a bad
  second rank, failed as intended and their workers were reaped. Unfiltered
  Compute Sanitizer memcheck attached to the real one-GPU rank and reported
  zero errors and zero bytes leaked. No performance claim is made. Provenance,
  the initial SIGINT harness failure, cost and limits are recorded in
  `test_results/refactor_20261114/STAGE_NATIVE_GRAPH_BINDING.md`.

## 2026-11-15 single-read native generator ownership
- The permutation runner now obtains generator bytes and names from one parsed
  file read, and borrows both vectors from the same owner. It retains the
  existing device transfer and old separate loader APIs. Bi-uniqueness checking
  no longer allocates a tree node per permutation index. This prevents pairing
  generator bytes and names from different reads; it does not bind the loaded
  file to the Rust graph hash or make concurrent file writes atomic.
- FIFO and format compatibility tests, Linux ASan/UBSan, the Rust workspace,
  643 Python integration tests (9 skipped), and four bounded one/two-RTX-3060
  public searches passed. A five-pair saturated full-step comparison had median
  +2.160% throughput, below the GPU speedup threshold; requested host-heap peak
  and uninstrumented loader time fell on the measured synthetic fixtures.
  Source-bound evidence and limitations are in
  `test_results/refactor_20261114/STAGE_NATIVE_GENERATOR_SNAPSHOT.md`.

## 2026-11-15 Rust permutation graph identity validation
- Rust `SearchSession` now recomputes the CayleyPy checkpoint SHA-256 from the
  ordered permutations and center before accepting an identity, including
  trivial requests. A stale hash returns an error and releases resident ranks.
  The digest is streamed without constructing another whole graph JSON buffer.
  This does not bind the file subsequently read by the native runner or apply
  to CNOT. Existing valid public graph hashes retain their format.
- Linux Rust workspace, release PyO3/worker, cancellation, and 643 Python
  integration tests passed (9 skipped). Four bounded public Rust-session calls
  on one/two RTX 3060s returned independently replayed paths and reused the
  expected resident PIDs. This is not a throughput or sanitizer result.
  Source-bound evidence, limitations and costs are in
  `test_results/refactor_20261114/STAGE_RUST_GRAPH_CONTRACT.md`.

## 2026-11-14 size-bound Rust MLP launch verification
- The existing default-OFF Rust MLP snapshot path now binds the previously
  validated manifest and tensor lengths to bounded streaming SHA-256 checks,
  including during copy and before launch. Legacy extensions use the full
  Python validation path. Session identity includes immutable graph contents,
  geometry and model lengths; graph changes release resident workers.
- Linux `cargo test --workspace --locked` passed with one ignored ABI fixture;
  the real PyO3 smoke and full Python suite passed (641 passed, 9 skipped).
  Ten isolated 2xRTX3060 sessions compared the same Rust snapshot owner using
  Python versus Rust launch verification. The median paired full-call change
  was -3.59%; the full-step promotion criterion was not measured. The option
  remains OFF.
  Source-bound receipts, precise scope, limitations and costs are in
  `test_results/refactor_20261114/STAGE_RUST_MLP_LAUNCH_VERIFY.md`.

## 2026-11-14 Rust-owned experimental MLP snapshots
- The opt-in strict-native MLP path can now delegate bounded snapshot copy,
  identity, staging and reuse to `beam-runtime` via PyO3, while Python retains
  model format and launch-boundary validation. Older extensions fall back to
  the previous Python owner. Linux publication uses `RENAME_NOREPLACE`; Unix
  input opens cannot block on a FIFO. Cooperative cancellation and unsupported
  network filesystem behavior are documented.
- Linux Rust workspace/build and the PyO3 smoke passed; Python reported 631
  passed and 9 skipped. Twenty bounded one/two-RTX-3060 cases and four mode
  transition groups passed. Independent validation compared 45 parameters in
  all 30 rank logs from 15 isolated measurement sessions and replayed paths.
  The synthetic five-pair median was +0.20% versus reuse OFF and +1.39% versus
  the Python snapshot owner. The experimental option remains OFF. Evidence,
  cost, and untested profiles are in `STAGE_RUST_MLP_SNAPSHOT.md`.

## 2026-11-14 opt-in MLP loaded-buffer binding
- Rank-owned host tensor buffers and the parsed manifest are hashed before GPU
  upload; each rank must supply its digest receipt before a result is returned.
  This closes the file-validation-to-loader identity gap for the experimental
  MLP/CUTLASS path, without adding another complete tensor copy. Its bounded
  tests and scope are in `STAGE_MLP_LOADED_BUFFER_BINDING.md`.

## 2026-11-14 snapshot validation revision: two-RTX-3060 result
- Commit `46c02ab5` passed 607 Linux tests (nine skipped) and 20 bounded GPU
  cases on one/two RTX 3060s with both session owners. Five alternating paired
  repeated full calls gave median 0.818708 s baseline versus 0.824459 s reuse;
  median pair change was +0.58%, with individual pairs from -1.72% to +4.53%.
  This is not an acceleration; the option remains default-OFF.
- The two evidence archives were fully hash-verified and samples restored;
  task-owned Vast lease 60648840 was destroyed and its absence confirmed.
  The elapsed GPU/disk estimate is $0.0376, excluding network charges; see
  `test_results/refactor_20261114/STAGE_MODEL_SNAPSHOT_REUSE.md`.

## 2026-11-14 snapshot verification pass and first GPU result
- The first two-RTX-3060 run passed 20 bounded Python/Rust session cases and
  605 Linux tests (nine skipped). With an 18,369,835-byte synthetic FP16 model,
  five paired repeats showed median full-call change +4.59%; the worker timer
  was nearly unchanged. The snapshot option stays off by default.
- Removed one duplicate retained-snapshot validation on cache hits. The source
  is still validated twice, and the actual retained snapshot is verified in
  `NativeSession.run` before submission or reload. A corruption-during-runtime-
  preparation test exercises that real session path. The revised option awaits
  a new GPU measurement; see `test_results/refactor_20261114/`.

## 2026-11-14 experimental session model snapshot reuse
- Added an explicit, default-OFF session option that reuses one private snapshot
  for a supplied MLP `NativeModel` in strict native mode. Source, retained bytes,
  graph/content identity and cache directory are checked on reuse. The session
  cache entry retains a path, and prior run evidence is not deleted. A worker
  relaunch snapshots again; only already-loaded compatible workers reuse it.
- A public search now reserves its session through model/runtime preparation
  and result assembly. `close()` is rejected during this scope, and `run()`
  rechecks closed state after acquiring its own lock. Direct entry points reject
  concurrent use, and failed direct launches release newly-created workers.
  Reload is bound to the exact prepared model; a contention test covers the
  interval before a competing owner is published. The existing file loader
  does not atomically bind its read to prior verification against concurrent
  file mutation; the transfer register now describes Python validation copies.
- CPU contract/regression checks are recorded under `test_results/refactor_20261114/`.
  GPU correctness and paired repeated-call performance are pending; this interim
  cache is Python-owned and does not complete the Rust preparation migration.

## 2026-11-14 public index sort: two-RTX-3060 validation
- The public ON/OFF selection passed 28 bounded session cases per backend on
  one and two RTX 3060 GPUs, with Python and Rust session owners. Linux tests
  reported 577 passed and nine skipped; the Rust workspace tests/build passed.
- Five alternating repeated calls had median wall times 0.414 s OFF and
  0.394 s ON; median paired change was -2.74%. This is a synthetic public-call
  fixture, not an isolated saturated-step or per-mode peak-VRAM measurement.
  The option remains experimental. Results, source/binary checks, provider
  release receipts and limitations are in `test_results/refactor_20261114/`.
- Extended the transfer register with the per-call private `NativeModel`
  snapshot and the session-launch runner/NCCL copies. Snapshot reuse is not
  implemented; the existing content checks remain mandatory.

## 2026-11-14 public experimental Stream 4 index-sort option
- Added explicit `NativeOptions(experimental_index_sort=True)` wiring for the
  existing CUDA index sort, with separate managed build identities, runner and
  native-library mode checks, separate shared-profile compatibility, and typed
  Rust configuration. The default and
  the other CUDA experiment switches remain unchanged. The OFF Rust descriptor
  remains readable by the earlier extension.
- CPU contract/regression evidence and outstanding hardware checks are in
  `test_results/refactor_20261114/STAGE_PUBLIC_INDEX_SORT.md`.
- The first remote Rust suite passed. Its full Linux suite exposed an obsolete
  mocked build receipt, now corrected; driver 535 could not initialize the
  CUDA 12.8 container, so GPU validation moved to a compatible host.

## 2026-11-14 two-RTX-3060 NCCL identity validation
- Linux regression: 570 passed, 9 skipped. CUTLASS and LibTorch public-session
  fixtures passed on one and two RTX 3060 devices, including session reuse and
  model/beam changes. Two setup failures are preserved with the successful retry.
- Five alternating pairs of complete repeated synthetic FP16 Hamming Rust calls
  had median wall times of 0.580 s baseline and 0.428 s with the experimental
  NCCL option; worker medians were approximately equal. The option remains
  experimental: this measurement is not an isolated GPU-step benchmark or
  acceptance for other models or accelerators. Evidence and limits are in
  `test_results/refactor_20261114/STAGE_NCCL_IDENTITY.md`.

## 2026-11-08 experimental Rust resident session
- Added explicit `NativeSession(engine="rust")` routing for whole searches to
  the Rust rank/session owner and native C ABI library. The existing default
  session, model preparation and calibration remain unchanged. This opt-in
  requires a managed build and matching packaged Rust artifacts.
- The Python suite passed 546 tests with nine skips; one- and two-RTX-3060
  public MLP session fixtures passed with CUTLASS and LibTorch. Five paired
  repeated public calls showed no material speedup. The installed wheel also
  passed a two-GPU session smoke test. A single host profile identified NCCL
  identity hashing as the dominant repeated-call preparation cost. Details
  and limits are in `test_results/refactor_20261107/STAGE5_PUBLIC_RUST_SESSION.md`.

## 2026-11-08 reject incomplete PyTorch-dependent runner metadata
- The on-demand native library builder now requires a PyTorch build identity
  for LibTorch, transformer/ensemble, and calibration runner configurations.
  Existing identity comparisons continue to detect changed PyTorch dependencies.
- The scoped CPU, packaging and 1/2-GPU session evidence is recorded in
  `test_results/refactor_20261107/STAGE5_NATIVE_TORCH_IDENTITY.md`.

## 2026-11-08 resolve the native library backend from the verified runner
- On-demand library preparation now accepts the default `auto` inference
  setting and uses the verified runner's resolved CUTLASS or LibTorch backend.
  Explicit mismatches and unresolved runner metadata still fail before build.
- See `test_results/refactor_20261107/STAGE5_NATIVE_AUTO.md` for CPU and GPU
  test scope. This does not switch public search to the Rust owner.

## 2026-11-08 lazy native library build and ABI inspection
- Added an explicit on-demand CUDA shared-library builder for a compatible
  managed runner. It reconfigures the managed CMake options before building,
  verifies dependencies and binary hashes, and inspects the actual ABI layout
  in a separate Rust worker. Cache hits validate the receipt and binary again;
  CUDA is not loaded into the Python process by this operation.
- The preparatory builder is not connected to the public Python search path.
  Hardware and test scope are recorded in the associated `test_results` report;
  no end-to-end performance claim or unsupported hardware acceptance is made.

## 2026-11-08 package and discover the Rust worker
- Wheels now build and ship `beam-runner` from the same bundled Rust workspace
  as the PyO3 extension. PEP 517 builds stage outside the checkout; editable
  installs put generated worker/extension artifacts in the ignored package
  directory. The adapter discovers an executable worker without loading CUDA.
- Checkout, source-distribution, isolated and editable installs passed. On one
  and two RTX 3060 devices the worker matched a previous runner across repeated
  searches. Rust SearchSession also passed model/beam changes, collection,
  cancellation and injected rank failure in a bounded fixture with CUTLASS and
  LibTorch native libraries. These checks do not switch public orchestration or
  establish a search-step performance gain.

## 2026-11-08 Rust Python distribution and matching native sources
- Replaced the setuptools-only wheel with a PEP 517 maturin wrapper that stages
  the matching native tree without modifying the checkout. Wheels include the
  ABI3 PyO3 extension and both public package names. Source distributions retain
  and verify the bundled native-source SHA-256 manifest before rebuilding.
- On the owned 2x RTX 3060 machine, checkout and sdist wheels had identical
  329-file native manifests; isolated and editable installs passed. The installed
  package passed 528 Python tests (9 skips), plus 33 focused CNOT CPU/CUDA tests.
  A newly built bundled-source CUDA library and installed extension passed repeated
  Rust/native reference sessions on one and two physical GPUs. These checks do
  not switch the public Python orchestration or establish performance gains.
- The wheel does not yet bundle a ready `beam-runner` or CUDA library; artifact
  discovery/building and public runtime switching remain migration stages.

## 2026-10-05 captured GEMM launch fields observation
- Actual new native typed graph probe RED missinglaunchvalues, then collector
  exposes tiled grid/swizzle/k-size/semaphore-null plus universal mode,batch
  count and D-stride. Native regular+broadcast baseline/mutations pass both
  RTX3060 under hostUBSan. Synthetic kernels never launched. Independent
  expected-value gate and freshly rebuilt trained runner remain pending.
  Previously preserved76f4runner does not emit these new fields; do not reuse
  its earlier numerical evidence to claim current full parameter coverage.

## 2026-10-05 fresh trained per-GPU numerical/layout evidence
- Restored immutable runner76f4e95e... and retained prepared reference weights;
  current-source CMake probe build pinnedCUTLASS, new supervisor on owned54269617.
  Fresh two physical-device candidates pass raw numeric/ranking comparisons,
  GEMM parameter layout/transfers/dependencies/scalars/pointers/dims/network/
  attention checks. ExpectedCLI2 numeric_pass_unadmitted, not a beam campaign.
- Full physical function argument offsets/sizes independently checked on both
  fresh graphs:6nonGEMM+1attention+4GEMM functions each. Probe bundle unchanged
  before/after. Complete parameter VALUES/admission remainfalse. Profile32/13x2,
  not1024x4 performance/target-H200 acceptance. Fresh artifact archive created.

## 2026-10-05 supervisor SM80 probe wiring
- Actual bundle-loop RED missingSM80table; now observes defaultSM75 and explicit
  --sm80 from same hashed binary, requires correct output scope and unchanged
  binary between invocations. Actual preflight-selection RED omitted table on
  SM86; now selectsSM80table forSM80/86, preservingSM75 and unsupportedSM90.
  No production_admitted promotion. Fresh hardware graph still pending.

## 2026-10-05 configured archived regression and complete argument layouts
- ExplicitCUTLASS_DIR nativeCMake focused1passed. Configured archived source
  suite922passed/108skipped/28subtests196.96s, public subtree explicitly excluded.
  Archived source plus current ABI overlays, not every currentcheckout edit.
- Independently compiled attention size table plus recorded physicalSM86 graph
  validates all function argument offsets/sizes. Extended5tests reject mutated
  residual argument size/offset/signature. Parameter values/admission remainfalse.
  Historical capture replay only, not fresh GPU/H200 acceptance or full goal.

## 2026-10-05 real recorded graph route correction
- Complete recordedSM86 graph RED exposed incorrect new SM80 descriptor metadata.
  Corrected residual3 and included SM75pipelined2 head as separate record; table
  now5entries including split bias. Underlying runtime kernels not modified.
  Independently compiled probe rebuilt remotely; complete validator on archived
  physical graph plus focused testsGREEN19. Not fresh graph or full admission.
  Archived-project regression dispatched remotely with public subtree excluded.

## 2026-10-05 table/device architecture binding
- Actual extracted graph-layout table branch RED2failed/3passed: SM86 rejected
  and SM80 table improperly accepted onSM75 after parser expansion. Bound scope
  strictly toSM75 versusSM80/86; unsupported architectures remain rejected,
  includingSM90 pending target route proof. Actual remoteGREEN18focused tests.
  This verifies the table-binding branch, not whole captured graph or fullsuite.

## 2026-10-05 independently compiled SM80 argument sizes
- Extended standalone storage probe with explicit --sm80 selection, retaining
  defaultSM75. Actual old binaryRED1fail; new nvcc sm86 binary and pinnedCUTLASS
  0b55a2f691d69981583568fd9eb69687b1f0de8a GREEN13focused tests on owned54269617.
  CompiledParams sizes368regular/440broadcast including split,align8. Upstream
  sm100 defaulted-function annotation warnings retained in build log. No kernel
  execution, complete captured-node validation, whole suite or admission claim.

## 2026-10-05 remote SM80 storage parser RED/GREEN
- Owned54269617 SSH ready, both12GiB3060 observed; remote docs read. Installed
  pytest in remote venv only. Actual pending test RED1fail/4pass on old parser.
- Extended argument-storage parser with explicit complete SM80 descriptor set:
  regular pipelined2, broadcast3 and distinct full/split bias shapes. SM75 scope
  unchanged; reject duplicates/incomplete/unknown descriptors and invalid types.
  Remote focused suite12passed including oldSM75 tests. This is synthetic parser
  coverage, not physical compiled ABI or full regression/production admission.

## 2026-10-05 new10USD remote SM80 window
- User authorized a new bounded10USD cheap3060 lease. Added ownership-scoped
  single-attempt orchestration with2h stop deadline; no blind create retry.
  Previous20USD billing remains unresolved, not rolled into new authority.

## 2026-10-05 pending SM80 ABI validator regression
- Prepared explicit remote-only RED test in evidence/pending_tests for the
  SM80 argument-storage route, including distinct full/split bias descriptors.
  Synthetic ABI sizes are parser fixtures, not physical production evidence.
  Test not executed; no RED/GREEN claim or production validator change. Existing
  collector supports SM80 members; argument-size validator remains SM75-only.

## 2026-10-05 billing reconciliation probe
- Added read-only official charges-endpoint schema probe. Outputs only HTTP
  status, schema and pagination metadata, not credentials or unrelated charges.
  No rental/restart authority inferred from a successful response. Cumulative
  spending remains unreconciled until ownership and all pages are checked.

## 2026-10-05 authorized retired rental cleanup
- User explicitly instructed via coordinator to remove unneeded Cube4 leases.
  Verified original ownership reports, stopped live API state, preserved source/
  binary pins and scaling raw/native-status files before irreversible deletion.
  Deleted54044187/54150123/54154002/54161582; fresh API all four absent, only
  current54201325 retained. No restart, new rental or other-owner mutation.
  Disk contents not archived (preallocated history arenas) are unrecoverable;
  needed source/results retained. Receipt retired_cleanup_20261005.json.

## 2026-10-04 diagnostic profile isolation correction
- Actual strict p10 run found solution but terminal publication rejected stale
  rank-status reservation inherited from earlier benchmark. Helper also forced
  inner13 despite supplied256. Failed run retained, not accepted as profile test.
- Both original/reflected helpers now own fresh absolute rank-status paths and
  preserve supplied micro (legacy13 only if absent). Actual extracted helper
  RED2original+2reflected -> GREEN4. No production fail-closed status weakening.
- Fresh current1024/256x4 originalp10 strict two-rank/replay passes,length9;
  reflected and final regression in flight. Goal/admission remain incomplete.

## 2026-10-04 physical larger-batch acceptance
- Same timingOFF runner2RTX3060,beam65536,depth_done8:1024outer/256inner/4lanes
  completes two repeats0.852651/0.852501s, native two-rank status/depth/immutable
  input checks pass. True preceding parent frontier65536 ->76.9k parents/s.
  Approx91.16% of sequential isolated micro256/lanes4 summed84.3k reference;
  standalone inference/chunk grouping differ, no positive numerical admission.
  Profile improvement~2.4x versus32outer/13inner/2lanes, not code-only speedup.

## 2026-10-04 timing-off throughput controls
- Actual CUDA runner confirms timingOFF/depthlogsON/debugON, fixed source built.
  Six sequential isolated Stream1 GPU rows measured micro128/256/512,lanes4;
  micro256 about42k parents/s per3060. Not exact production chunk admission.
- Same-binary control two-rank65k/depth9 outer32/inner13/lanes2 completes both
  times; depth8 seconds2.03517/2.10756, globalfrontier65536. Repeat labels are
  not code before/after; no cross-configuration Stream1 efficiency inference.
- New1024outer/256inner/4lanes profile increases staging slots/capacity without
  global-search semantic change; real validation launched on existing lease.

## 2026-10-04 optional pipeline timing parser
- Remote literal release log reproduces KeyError(stream4_ms); debug row passes.
  Parser now retains absent timing/queue fields as None, not fabricated zeros.
  Focused parser plus actual build-info branch6passed remotely under UBSAN.
- Actual timing-off CUDA runner built, but debugOFF forced depthlogsOFF despite
  requestON. Observed binary flags expose misconfiguration; reconfigure debugON,
  timingOFF/depthlogsON. No performance result claimed from first binary.

## 2026-10-04 effective build-info flags
- Actual extracted runner --build-info branch: four remote RED failures for
  missing effective timing/depth/debug fields, then4GREEN under host UBSAN.
  Added boolean fields to actual branch; search algorithm/ABI unchanged.
  Full CUDA binary/regression pending, do not infer hardware acceptance.
- Owned release lease54201325 at0.1177777778USD/h,2h stop guards within existing
 20USD authority. Cloud invoice endpoint401; cumulative actual billing unknown.

## 2026-10-04 build timing provenance regression preparation
- Added executable extracted actual --build-info branch regression for all
  four timing/depth-log combinations under host UBSAN. No fabricated runtime
  receipt and no local execution. Remote RED still required before modifying
  runner; not recorded as passed or fixed. Current cloud inventory confirms
  no active test rental and prior own54195079 absent.

## 2026-10-04 Stream5 source attribution correction
- Confirmed major count/request/payload/remote-collect synchronization outside
  timing guards; withdrawing timer-only attribution. Prior outer32 correctness
  profile is not throughput tuning evidence. Binary build-info lacks effective
  timing flags and paired helper requires timer-only fields; release comparison
  needs provenance/parser corrections before execution. No production edit or
  fresh hardware claim. Evidence: STREAM5_SYNC_SOURCE_AUDIT_20261004.md.

## 2026-10-04 blocking Stream4 slot HOL correction
- Confirmed on both remote RTX3060 with real CUDA event skew: blocking fallback
  waited about297ms on the oldest slot while a later slot was ready. Native
  extracted-source test RED before production edit, two failures as expected.
- Blocking multi-slot fallback now uses the existing all-slot release callback
  until any completion; preserves single-slot/full-drain synchronization and
  logical-shard/threshold semantics. No Stream4 algorithm or ABI redesign.
- Fresh owned lease54195079,2h guards,within existing20USD; previous own lease
  deleted. Native GREEN7passed and four actual two-rank dispatcher modes clean
  under memcheck/full leaks. Alternating saturated65k beam median2.06954 versus
  2.07853s: no meaningful throughput gain. Separate actual Nsight traces saved;
  timing instrumentation confounds release-path synchronization attribution.
- Fresh original/reflected1..10 physical cohort replay/depth reconciliation and
  40 native statuses pass; asymmetric rank1 failure control passes. Archived
  project regression910passed/108skipped/28subtests; missing public-subtree
  configs excluded explicitly, initial failed collection/configure retained.
- Final source/evidence/log archives hash-matched locally (evidence426e4457...);
  provider confirms own54195079 stopped/deleted/absent20:43:41UTC. Other leases
  untouched. Whole goal still active; remaining admission/scale gates explicit.
- Evidence: `test_results/audit_fixes_2026-09-28/PIPELINE_54195079.md`.

## 2026-10-04 current canonical-fusion checkpoint
- Final remote suite1175passed/106skipped/28subtests. Raw final archive
  e5c07a756d2ad18f488a4936ce62efe3e7164ec21176652d8ad652b11abc2146
  downloaded/hash-matched; extracted runner identity verified. Provider confirms
  stop/delete/absent record for own54186691; other instances untouched. Complete
  positive admission and H200/full-depth gates remain open.
- Fresh complete two-rank original/reflected1..10 cohort passed independent
  replay and depth reconciliation;40 native solved terminal statuses matched.
  Added missing VERBOSE switch to diagnostic helpers (first series retained
  as replay/memcheck evidence, not depth-trace acceptance).
- Asymmetric rank1 finite overflow rejects both ranks without result/status
  publication; puzzle1000 beam4096/depth8 completes unsolved on both ranks with
  exact8-row native depth traces. Strict memcheck clean for both controls.
- Mandatory A/B integration and actual fresh scorer verification are now
  complete for the bounded six-signature scope: 48 protocol/profile assertions
  per GPU, zero strict scorer memcheck errors/leaks and 48 native metadata
  mutations per GPU under host UBSAN. Complete production admission stays false.
- Ignored input-fusion alias corrected in diagnostic scripts; old captures are
  unfused evidence. Current actual long-key profile independently checked.
- System NCCL failed strict two-rank sanitization; isolated NCCL2.25.1 passed
  puzzle1 unchanged runner, original length2/replay valid. Paired cohort running.
- Hash-matched local archive and detailed evidence in
  `test_results/audit_fixes_2026-09-28/CANONICAL_FUSION_54186691.md`.

## 2026-10-04 private GEMM A/B typed observation
- Continuing20USD authority restored remote work without per-launch asks.
  Own54186691 2xRTX3060,0.11778USD/h,2h independent/server guards;no local tests.
- Native A/B metadata producer REDmissingvalues,then eight base-field mutation
  observations GREEN on both GPUs. Exact pinned standard-layout wrapper chain
  and size/alignment assertions; no private guessed offsets or CUTLASS edits.
- Independent literal mainloop math23RED->GREEN. Compiled descriptor and exact
  private-layout header SHA binding RED->GREEN; combined34remote tests passed.
- Fresh native runner built; initial pending checks superseded by current
  canonical-fusion checkpoint above. No admission promotion.

## 2026-10-04 mandatory public GEMM iterator binding
- Collector mutation protocol RED28 failures, then saved native capture replay
  GREEN28/28 on each GPU dataset. Validation binds device/execution/inventory,
  exact independent compiled geometry and bounded public C/D iterator fields.
- Actual remote probe-bundle RED exposed omitted iterator table. Added CMake
  target and bundle/source/binary identity; graph CLI now requests this check.
  Actual mandatory bundle GREEN1/1 in26.52s. First full run1151passed/80skipped
  with one missing CUTLASS_DIR environment failure preserved. Corrected full
  rerun1152passed/80skipped/28subtests in222.33s,exit0.
- Fresh native preflight on GPU0/1: eight raw jobs128x24 scores each,numeric
  violations0,top1/top4 agreement1.0,public iterator flagtrue on both. Exit2,
  numeric_pass_unadmitted and production_admittedfalse retained deliberately.
- Fresh-capture collector protocol28/28passed per GPU. Same fresh runner strict
  unfiltered memcheck/full leak checking0errors/0leaks on each device,exit0.
- Source/evidence saved locally,archive and binary SHA256 verified. No local
  build/test/inference; no hot-path or Stream4 architecture changes.
- No private A/B,split,activation/order or full production admission claim.
- Own54182164 stopped/deleted18:28:27UTC after verified local source/evidence
  archival; authenticated record absent. Approximate35m41s advertised cost0.0893USD
  plus traffic,not account invoice. No remaining paid instance from this stage.

## 2026-10-04 unsplit QKV/FF1 bias values
- Independent literal QKV tests16cases observed remote RED missing validator.
  Added bounded per-node shape,layer-specific weight/bias,A/C/D role/extent,
  alpha/beta,batch stride and null tensor validation. No row iterator Params,
  epilogue activation/order or split/head admission claim. GREEN pending.

## 2026-10-04 residual GEMM independent values
- New per-node residual validator15cases RED then remote GREEN0.06s: exact
  layer weight role,M/N/K,alpha/beta,strides,A/C/D roles,extents,null aux fields.
- Added actual captured residual count/value test; whole graph ordering and
  bias/ReLU/head/split GEMM values remain separate open requirements.

## 2026-10-04 GEMM producer native GREEN
- Remote retry build succeeded; fresh trained GPU0 scorer and typed GEMM
  producer coverage test1/1passed0.04s on baseline compact/finalCLS profile.
  Independent M/N/K/stride/matrix-role validator and mandatory GEMM gate open.
- Local source archive captured_params_source_54161582.tar.gz SHA256
  e48cc373569d4dc09af00777c6012bff62a7b33bd31fe36f216070ca485d8ee2.
  Real score/build/RED logs preserved in gemm_producer_evidence archive.

## 2026-10-04 GEMM broadcast specialization build repair
- Native build rejects nonexistent C1/C2 fields: selected true broadcast
  specialization uses ptr_C/batch_stride_C, unlike unspecialized header branch.
  Corrected exact selected typed access; no guessed offset/ABI-size cast.
  Remote retry pending; keep failed log and no producer GREEN claim.

## 2026-10-04 captured GEMM typed producer implementation
- After native producer RED added exact CUDA-function-name matching typed
  regular/broadcast baseline Sm75/Sm80 and fixed split-QKV Params extraction.
  Copies actual argument0 after compiled ABI check; exports problem, pointers,
  alpha/beta and strides. Unknown specializations explicitly remain unknown.
- Registry now includes actual loaded weight/bias allocation byte sizes.
  Diagnostic-only; remote rebuild/producer GREEN still pending; no admission.

## 2026-10-04 captured GEMM producer RED
- New actual-scorer opt-in GEMM typed member producer gate observed RED1/1
  0.07s: graph_gemm_members.json absent. This is missing evidence, not a proven
  wrong GEMM execution. Native numerical candidates remain bounded prior evidence.
- Pinned CUTLASS regular Params uses ref_A..D; broadcast fused Params uses
  ptr_A/B/C1/C2/D/Vector/Tensor and batch strides. Separate exact typed decoding
  required; do not reinterpret by matching size or requested policy text.

## 2026-10-04 actual attention collector protocol
- Mandatory integration full remote regression1091passed/17skipped/
  28subtests passed221.35s. Subsequent opt-in protocol tests not in that run.
- Real captured split artifacts replayed via owned subprocess collector11/11
  passed1.20s: missing/device/hash/stride/short/role/lane/node/promotion reject;
  valid original returns graph_attention_members_checked but never admission.
- These are protocol tests, not new native inference; raw logs saved locally.

## 2026-10-04 native attention Params mutation evidence
- Synthetic graph native typed Params probe mutates actual kernel-node arguments
  with cudaGraphKernelNodeSetParams; both RTX3060 pass: stride/role/short values
  remain observable, unregistered pointer rejects. Not trained-model replay.
- Actual captured split report mutations15/15passed0.30s: lane/node/function/ABI
  binding, scalar/pointer/scope/admission changes reject. These test reports,
  not model quality. Logs saved; latest full regression still pending.

## 2026-10-04 mandatory attention split preflight verified
- Pointer+geometry focused50passed0.47s. Fresh actual main preflight on both
  RTX3060 split-QKV32/13/two-lane now passes scalar/Dims/pointer/NetworkView
  and mandatory captured attention member values, raw numeric candidates pass.
- Remains numeric_pass_unadmitted; opaque GEMM Params/complete positive startup
  admission not established. Latest full suite started remotely; no H200 claim.

## 2026-10-04 split query gather pointer profile
- Independent pointer profile also omitted split query gather from lane_context
  to lane_attention. Literal role/extent test observed remote RED; expectation
  added without changing native inference. Focused and actual preflight pending.

## 2026-10-04 split-QKV gather expectation defect
- Real mandatory preflight rejected embedded Dims: independent expected3gathers
  versus actual6 per lane. Split-QKV query input needs extra gather per chunk.
- New independent profile test observed RED3!=6; geometry expectation corrected
  to include split query gather. Native inference unchanged. Pointer/scalar
  expectations and fresh main preflight/full regression still require reconciliation.

## 2026-10-04 mandatory attention member collector
- Fresh GPU1 padded64/full/two-lane and padded57/CLS-split/two-lane artifact
  independent values/counts passed1/1 each. No performance/admission claim.
- Mandatory collector missing-sidecar case observed RED; independent typed
  member contract + content/device bindings integrated into main preflight.
  Focused40passed2.46s; prior full1088/17/28passed224.37s predates integration.
- Fresh actual mandatory two-device split-QKV preflight started; result pending.
  Full regression and native Params mutation probes still required.

## 2026-10-04 captured attention producer GREEN/value validator
- Remote typed native build succeeded; fresh GPU0 scorer producer test1/1pass
  0.04s. Captured Params values now observable, not yet production-admitted.
- Independent Cube4 FP16 per-node value validator:14remote cases RED missing
  implementation then GREEN0.07s. Reject scalar/role/offset/short/null mutations.
- Native artifact test additionally validates every attention node and complete
  full/CLS/chunk multiset. Actual-value result pending; mandatory collector,
  native parameter mutation and full suite still needed.

## 2026-10-04 typed attention Params extraction pending build
- After real producer RED, added diagnostic extraction by exact compiled CUDA
  function name of12 supported AttentionKernel specializations, Params ABI size
  checked before memcpy. Captured pointers resolve against actual registry;
  publishes scalar fields and null endpoints, bound to same graph enumeration.
- No normal launch change; no independent value-validation/admission claim.
  Remote production_runner rebuild started bounded600s on own54161582.

## 2026-10-04 attention Params producer RED
- Complete remote regression after diagnosis fix1074passed/16skipped/
  28subtests passed221.18s. Subsequent attention opt-in test not in that run.
- Added actual scorer attention member producer contract, opt-in remote only;
  observed RED1failed0.07s because native graph_attention_members.json absent.
  This is missing evidence, not a demonstrated bad attention pointer/numeric bug.
- Next requires typed captured Params extraction, independent value/stride/extent
  validator and native mutations before mandatory collector integration/admission.

## 2026-10-04 baseline/multi-lane native candidate coverage
- Fresh real main CLI on both3060 with outer32/inner13/lanes2,padded64,
  fused-input0/final-CLS0 also passed raw-score and consumed NetworkView checks.
  Covers independent input_graph branch and partial chunks; remains unadmitted.
  Full regression after diagnosis change still pending; no local compute.

## 2026-10-04 real dual-device preflight and diagnosis
- Fresh two-device preflight raw-score comparisons and12 graph checks per device
  passed including mandatory consumed NetworkView members. Remains unadmitted;
  subprocess returned nonzero, zero production rank startup attempted.
- Incorrect missing_admission label claimed absent launcher integration despite
  mandatory launcher call. New regression RED confirmed misleading diagnosis;
  replace with missing complete independently validated production admission.
  No admission policy or return-code change.

## 2026-10-04 full NetworkView regression GREEN
- Remote bare pytest with explicit CUTLASS_DIR completed:1073passed,15skipped,
  28subtests passed221.27s. Actual native opt-in checks separately27/27passed.
- Fresh two-device main production preflight running on own54161582; expected
  incomplete admission must remain exit2, never bypassed for search startup.

## 2026-10-04 second-device and native extraction mutations
- Fresh GPU1 scorer artifact test passed1/1 in0.06s, separate device binding.
- Synthetic typed CUDA Graph probe passed: correct members extracted, swapped
  registered roles and short extents preserved rather than relabelled/enlarged,
  unregistered member rejected. These are extraction tests, not model numerics.
- Added opt-in remote pytest build/run harness. Existing25protocol tests cover
  collector rejection of invalid roles/extents. Full suite repeat started with
  explicit CUTLASS_DIR; result pending. All compute remains remote.

## 2026-10-04 native NetworkView evidence implementation
- Fresh remote scorer ran; explicitly enabled real-artifact test failed because
  graph_network_members.json was absent (native RED, not fixture inference).
- Diagnostic inventory now copies captured typed NetworkView argument4 and
  resolves consumed pointers against actual loaded tensor allocation spans.
  Publishes bound sidecar; main collector requires independent network profile.
- Remote rebuild succeeded; fresh GPU0 scorer and enabled native artifact test
  passed1/1 in0.07s. Native captured-pointer mutation probes remain unverified.
  Opaque Params and production admission remain open.
- Full remote pytest:1072passed/15skipped/1failed due missing CUTLASS_DIR in
  CPU-test subprocess. Targeted repeat with explicit CUTLASS_DIR:6passed108.81s.

## 2026-10-04 network protocol remote GREEN
- Own replacement54161582 authorized10USD,0.143USD/h; fixed19:45Moscow stopguard.
- NetworkView collector protocol regression25passed in2.17s on remote POSIX.
- Bare remote pytest started bounded600s; no full-suite/native admission claim yet.

## 2026-10-04 real scorer network evidence test prepared
- Added opt-in test_native_network_members.py: actual scorer output must contain
  NetworkView member sidecar with exact content/device bindings and independent
  Cube4 expectations. Missing sidecar fails once explicitly enabled, not skipped.
- Not run. This is artifact validation, not native mutation or replay admission.

## 2026-10-04 embedded network corruption coverage
- Expanded remote subprocess protocol tests from9 to25 cases: missing/duplicate
  lanes, nodes and members, swapped function identity, invented opaque coverage,
  offsets, boolean integer impostors, and scope/admission promotion.
- Static native trace: GraphKernelInventory currently copies NetworkView only
  for embedded dimensions; GraphPointerInventory explicitly excludes members.
- Native member extraction and remote verification remain open; no GREEN claim.

## 2026-10-04 consumed NetworkView protocol correction (verification pending)
- Re-read saved nine-case remote RED evidence: required embedded pointer profile
  was silently ignored by collect_fresh_candidate.
- Added graph_network_contract.py and collector binding/coverage/role/extent checks.
- Scope remains FP16 Cube4 known fused/input graph members, not opaque Params,
  native extraction or production admission. No tests executed locally.
- GREEN and full remote regression pending; stopped GPU was not restarted.

## 2026-10-04 controlled own RTX 3060 scaling
- Completed same-host world1/world2 puzzle1000 depth8, global beam262144.
- Step6/7 speedup 2.032x/2.024x; world2 about97% summed isolated Stream1.
- Both ranks exit0, UNSOLVED; local raw logs SHA256 match remote originals.
- Evidence: test_results/audit_fixes_2026-09-28/TWO_GPU_SCALING_54154002.md.
- No H200, full frontier identity, or production admission claim.

- 2026-10-04: Saved1GPU logs/current binaries with matching hashes, stopped
 54150123 and verified Inactive. Sequential own2GPU rental54154002 at0.168USD/h,
  stopguard RUNNING pid1245 deadline15:30:26UTC. Driver595.71.05 differs from
  old1GPU580.82.09; same-host world1 baseline required before world2 comparison.
  Pinned NCCL2.25.1 in isolated directory, no system-library replacement.

- 2026-10-04: Own1x3060 manual4-shard diagnostic cap131072 completes8waves;
  saturated7.91s at262144beam,33140parents/s,95.6% isolated. cap65536 stalls
  until bounded termination. Do not use exact logical-size physical capacity
  as generally safe; explicit headroom/runtime-progress evidence required.
  Raw logs saved. Not production admission/H200/scaling acceptance.

- 2026-10-04: Both target-GPU integration tests GREEN(2passed1.71s) after
  benchmark input and graph-addressing corrections. Valid Cube4 isolated256x2
 34647.2parents/s; baseline262144beam saturated24287-24676parents/s(70.1-71.2%).
  Full pytest running with known9 embedded-network REDs; no suite-green claim.

- 2026-10-04: Actual fixed-address NativeGraph runner regression RED confirms
  execution/addressing conflation. Chunk check now allows graph-fixed and
  graph-windowed while still rejecting eager-windowed and batch mismatches.
  Remote rebuild/regressions pending; no complete fix/admission claim yet.

- 2026-10-04: Fresh actual Cube4 default graph rejects non-windowed jobs due to
  executor/addressing conflation; supported window8 completes depth8 on1x3060,
  saturated steps10.62-10.79s at262144 beam. Not yet paired with valid Stream1.
- Benchmark input integration test RED remotely: explicit production-format
  input paths ignored. stream_benchmark now uses production native_input loaders,
  explicit BEAM_GENERATOR_PATH/BEAM_PUZZLE_INFO_JSON/BEAM_TEST_CSV and state checks.
  Rebuild/green verification pending; old timings invalidated as Cube4 evidence.

- 2026-10-04: Started own1xRTX3060 instance54150123 for sequential scaling.
  Existing public key verified enabled, no new grant. Remote-only SM86 rebuild
  uses source archive708debd73117cb8adff9a5f7506c78f74341ca5f8b3a7230b75124f45d443374.
  Stop guard RUNNING pid846; fixed deadline2026-10-04T15:01:41Z, no extension.
  No scaling/performance/production admission claim yet; other owners untouched.

## 2026-10-04 — Isolated old physical vs logical-union cost on owned two T4

New diagnostic tests/stream4_union_cost_benchmark.cu reuses parameterized test
allocation fixture, restoring identical inputs before each CUDA-event interval.
Capacity262144 per physical buffer, occupancy25/100%, cross-buffer overlap0/50%,
3 warmups+10 measured repeats per device. Old source81ed9f0 stream4 compiled
separately; no production code changes. Full-occupancy medians approximately
10.5ms for both two-old-physical passes and one logical-union pass. This is
NOT whole-pipeline slowdown: union is an additional final stage there.
Logs and exact limitations in STREAM4_COST_20261004.md. No H200 timing claim.

## 2026-10-04 — Consumed NetworkView gate regression prepared, not executed

Read original A01-A14 audit and current remaining admission plan. Confirmed
the seven consumed FP16 input-member extents against exact weight-loader
formulas: 9216/28672/512/512/512/336/168 bytes. Added nine fresh-subprocess
protocol regression cases in tests/test_embedded_network_supervisor.py.
No RED/GREEN or native extraction claim: tests have not been run, and no
production implementation changed. Local workloads remain prohibited.
New Vast IAB tab6 currently shows Login, not authenticated instance controls;
no restart or rental performed. Fresh additional USD10 authority recorded.

## 2026-10-04 — Owned54044187 stopped, renewed artifacts preserved

CLI stop acknowledged12:26:48UTC. SSH refused12:27:27UTC and fresh Vast UI
confirmed Inactive/Start/storage-only0.023USD/hour for exactly54044187.
No destroy/restart or other-owner mutation. Approximate base window0.55USD
from verified tariff/time, not final invoice; traffic exact invoice unavailable.
Final local closure report supersedes remote pre-stop pending copy; disk remains
paid/preserved. Current goal and full production admission remain incomplete.

## 2026-10-04 — Latest runner depth8 complete; renewed window preserving/closing

Latest7b516603... runner p1000 beam4096 depth8 completed UNSOLVED with both
terminal ranks and strict zero-error memchecks; archive5615fb9af3177487943501e6009bbdf872dc69587dd9dbb32bf612f975e24833
downloaded/hash matched. All new code/binary/log checkpoints preserved outside
Vast. WINDOW_CLOSE_RENEWED_20261004.md records scopes/remaining gates, tariff
math (not invoice) and pending manual stop; current goal remains incomplete.

## 2026-10-04 — Dedicated cross-buffer kernel/dispatcher receipts on2T4

Existing kernel_oracle/prefinal/collect/pipeline/pipeline_empty_rank fixtures
all passed serially after neural cohort cleanup. Nine strict zero-error rank/
process summaries, source hashes stable. Archived actual test binaries, sources,
requests and logs; download SHA05f4166c756ab043a335bc0169c7965ece59b0467035c57b52b791d045b1a3fc
matched. This is bounded synthetic uniform pipeline, not root p900 saturation
or neural/scaling acceptance. No shared/root implementation edits. Latest
production runner p1000 beam4096 depth8 now live with240s bound.

## 2026-10-04 — Fresh latest-binary cohort1..8; p9 timeout preserved

Same build-dims-sm75 original/reflected1..8 independently replay valid with32
strict zero-error rank summaries, lengths2/3/4/5/6/7/6/7 in both frames.
Original9 timed out180s; cleanup verified. Reflected9/pair10 not run, full1..10
not accepted; no algorithm/sanitizer defect inferred from timeout. Partial
accepted cohort and timeout archived separately, downloaded/hash matched.
No new instance or budget/stopguard extension; dedicated union tests are next.

## 2026-10-04 — Neighbor cross-buffer counterexample separated by branch

Received root p900/MLP120/128 scaling counterexample and independently verified
its downloaded archive SHA930b0aa95c913e740dff23a3c67aacc9c4e8a05f4d65057c9e19fc3caca17305.
Root dispatcher lacks final logical union; this Hopper worktree/remote runner
has it, hashes match639023f9.../aa6373ba.... No shared source fix or local replay.
Existing dedicated final-union dispatcher and kernel targets compiled SM75;
GPU execution queued only after serial neural controls. Added scoped handoff
report and corrected stale unconditional final-union-capacity wording in A08.
Regular shard/histogram capacity sorts remain; no performance closure claimed.

## 2026-10-04 — Actual embedded Dims CLI verified, still unadmitted

Separate remote build-dims-sm75 SHA7b5166032d861c7ec64a1ed2ec351775bde0b7a4f9ed192c14ba04815a39cc47
passes all11 graph checks on both T4 using sealed trained reference weights.
Actual CLI remains exit2/numeric_pass_unadmitted/production_admitted=false.
Stable six-source-hash full regression1048 passed/12 skipped/28 subtests,
186.77s. New binary two-rank p1 original/reflected verification pending.

## 2026-10-04 — Native embedded Dims producer and mandatory CLI binding

Added typed, size-checked captured argument copies for fused-input NetworkView,
gather and quantize Dims in the existing inventory observer; same native node
indices and function IDs. Sidecar binds device, execution hash and inventory
hash; CLI now requests independent Dims validation. Pointer members and opaque
parameter values remain outside this gate; full admission remains closed.
Native test RED is the missing dims_values_payload API with all required build
macros supplied (earlier include/config failures are not feature RED). Separate
remote build-dims-sm75 and actual two-device validation are pending. Previous
Python protocol suite:1048 passed,12 skipped,28 subtests,180.75s on remote T4.

## 2026-10-04 — Depth8 new runner complete; embedded Dims protocol RED/GREEN

New input runner puzzle1000 beam4096 depth8 completed unsolved with explicit
terminal ranks0/1 and both strict zero-error memchecks. Saved/downloaded/hash
matched. Seven fresh subprocess Dims tests first failed old supervisor, then
passed0.60s after independent fixed-model Dims validator/content/device bindings.
Profile requests now enforce the Dims sidecar, but actual C++ producer and CLI
mandatory profile wiring are not yet implemented. No actual captured Dims
acceptance or full production admission claimed. Full suite launched remotely.

## 2026-10-04 — Actual new input runner paired two-rank smoke

New input SHA b87c8e9a original/reflectedp1 passed2/2, independent aggregate
replay/input/binary/rank-depth binding and four strict zero-error rank memchecks.
Archived/downloaded/hash matched separately from quantize cohort. Launched
new-runner puzzle1000 depth8/beam4096 with500s timeout; terminal status/replay
acceptance awaits receipt. No large-beam or H200 performance claim.

## 2026-10-04 — Quantize cohort pair10 independently verified

Old quantize binary pair10 original/reflected passed9/9 with valid replay;
independent aggregate7..10 passes16 strict zero-error rank memchecks. Pair10
and aggregate source logs archived/downloaded/hash matched. New input binary
two-rank p1 smoke live54461, intentionally separate SHA and scope. No old
cohort evidence promoted to new binary or complete production admission.

## 2026-10-04 — Input full regression1041 verified

Latest remote suite1041/12/28 passed179.78s, terminalexit0, before/after hashes
match for six tracked source/test files. Logs/current edge tests downloaded and
hash matched. Input runner actual two-device score CLI passes; no actual new
binary two-rank beam receipt yet. Reflected10 old quantize runner live25358.
Full production numerical admission remains open; no H200 performance claim.

## 2026-10-04 — New input runner actual CLI passed; dual probe verified

Separate SM75 runner built SHA256
b87c8e9ae455be83147bf192c0e148cb6fd900900ce8d4f4aa5ba846badcd42b.
Fresh actual two-T4 CLI passes ten current checks including input top-level
pointer roles and nulls, exit2/unadmitted. Dual and nondual synthetic probes
replay39 on both T4s, strict0; required dual null rejects. Pointer edge suite44
passed1.38s, covering dual roles and padded64 logical57 grid. Full suite1039
passed188.81s predates these two added tests; latest immutable suite live78554.
Original10 solved length9/replay-valid on old quantize binary; reflected10 live.
Embedded Network/opaque Params/full admission remain open.

## 2026-10-04 — Input pointer independent contract GREEN; new build live

Independent literal input-role fixture first failed old validator (opaque
signature promoted). Added input role/extent expectations, explicit zero-extent
null validation and native fused grid binding. Forty-two pointer/supervisor
tests pass1.27s, including short frontier/wrong base/missing input/null mutations.
Separate build-input-sm75 runner/probe build live10588; full suite live35391.
Current cohort remains on unchanged quantize runner; reflected9 live25485.
No actual new-runner CLI or full production admission yet.

## 2026-10-04 — Input top-level pointer probe RED/GREEN, integration pending

Synthetic CUDA input-LN signature probe confirmed missing observer coverage RED.
Added eight top-level indices0,1,2,3,5,8,9,10, with null8..10 only for Dual=false.
Remote compiled probe replay39 passes strict memcheck0 on both physical T4s.
Registered actual diagnostic frontier/base allocation spans in score mode.
Independent Python input-role contract, null negatives and fresh actual runner
build/CLI remain pending; existing trained runner binary is unchanged. Embedded
NetworkView is not covered. Original9 solved length10/replay-valid; reflected9
live, sequential after input probe. No full production admission claim.

## 2026-10-04 — Latest immutable full regression1034 and pair8

Latest remote full regression1034/12/28 passed187.00s, terminalexit0 and
before/after six changed source/test hashes equal. Fresh scalar chunk CLI
ten current checks pass but admissionfalse. Pair8 aggregate7/7, strict0 each
rank/orientation. Saved source/logs/receipts archive downloaded/hash checked.
Original9 live62340; owned stopguard deadline remains approximately12:42UTC.

## 2026-10-04 — Three additional scalar chunk swaps RED/GREEN

Remote RED reproduced full/tail scalar swapping for fused input, CLS gather,
quantize. Added per-native-grid batch binding for those families. Focused34
tests pass1.09s and fresh two-T4 CLI ten checks pass; admission still false.
Earlier scalar full suite1031/12/28 completed180.78s but predates these three
tests/change, so new full suite launched with before/after six-file hash check.
Original8 solved length7 replay valid, reflected8 live62678; window unchanged.

## 2026-10-04 — Scalar geometry defect RED/GREEN and actual CLI

Confirmed same-symbol full/CLS scalar rows swap passed prior validator; remote
RED observed. Bound LN/bias-LN rows to actual native grid. Focused31 tests pass;
fresh two-T4 CLI ten current checks pass, admission remains false. Full suite
after change live69336. Pointer edge suite37 passes; preceding full1030/12/28
predates scalar fix. Puzzle7 paired independent aggregate6/6, four memchecks0.
Archive downloaded/hash matched. Puzzle8 original live2112. See
test_results/audit_fixes_2026-09-28/SCALAR_GEOMETRY_20261004.md.

## 2026-10-04 — Geometry full regression and fresh actual CLI verified

Remote full suite1020 passed/12 skipped/28 subtests175.43s. Fresh per-device
T4 CLI passed all ten current graph checks with geometry-binding mandatory;
intentional exit2/numeric_pass_unadmitted, not full production admission.
Reflected6 passed length7, both strict memchecks0 and original-frame replay.
Sources/logs/receipts archived and downloaded as geometry_verified_20261004.
Launched original7 sequentially with300s harness limit; cohort7..10 remains open.

## 2026-10-04 — Fresh authorized remote window; pointer GREEN

Resumed same54044187 at approximately10:42UTC under fresh user $10 ceiling.
Managed7200s stopguard RUNNING; other owner instance untouched. Uploaded four
geometry patch/test files. Remote focused suite27 passed1.06s. Full regression
and reflected6 are live; actual fresh CLI verification remains pending.

## 2026-10-04 — Pointer geometry binding implemented, verification pending

After the remote RED reproduction, added optional per-native-node grid binding
to the independent pointer role validator. Production preflight always supplies
captured grids; full-token/CLS role swaps can no longer match solely by multiset.
Supervisor fixture now uses hand-derived actual row grids rather than all ones.
No local workloads run. GREEN/full regression and fresh actual CLI acceptance
remain pending remote execution. Same-row block ordering and opaque parameter
values are not covered by this change; production admission remains false.

## 2026-10-04 — Puzzle6 original saved; remote window closing

Latest quantize binary solved original6 length7 with valid replay. Downloaded
and SHA-256 matched its evidence archive; reflected6 remains untested. Saved
pointer geometry RED reproduction; implementation remains pending. Authenticated
stop request for owned54044187 acknowledged before unchanged two-hour deadline;
completed shutdown and final invoice require external verification. See
test_results/audit_fixes_2026-09-28/WINDOW_CLOSE_20261004.md.

## 2026-10-04 — Latest puzzle5 pair accepted, puzzle6 running

Reflected5 completed length6 with independent inverse original-frame replay;
aggregate original/reflected5 verifier passed, four strict memchecks zero errors.
Pair archived and downloaded/hash checked. Latest controls1..5 now have bounded
pair evidence across original cohort and diagnose…162831 tokens truncated…ly (a56f59f). Current fullfrontier sweep remains live session9959 driver4287; first four baseline depths about36.74-36.76s each. No completed profile selection yet.

2026-10-09: Tightened rank-plan admission: native LOCAL_RANK communicator IDs must cover each rank exactly once when present; repeated node-local CUDA ordinals remain valid. Added128-rank sixteen-node cohort regression and rejected negative memory estimates. No128GPUhardware verification implied. Actual profile sweep continues with already-imported candidate selector, avoiding benchmark restart.

2026-10-09: Added bounded graph-derived calibration frontier generator: legal randomized walks with mixed lengths/parities, exact unique states across all ranks, excluded solved/direct-predecessor states, zero native padding and SHA256 receipts. Explicit state/time/attempt budgets reject incomplete frontiers instead of timing repeats as full work. Small finite graphs can decline frontier calibration without affecting normal search. Deterministic/reachability/padding/budget tests published, remote verification pending. Public pipeline hook still pending.

2026-10-09: Wired automatic downstream ensemble calibration into run_native after private runner/NCCL snapshot: requested-beam all-rank admission for every candidate, bounded graph-derived legal fixture, inference microbatch frozen, statistical full-step selection, final re-admission with reserved time, and explicit measured vsrequested beam metadata. Newoptions pipeline_seconds600/frontier_maxstates65536; unfillable small graphs retain ordinary native search and report pipeline notmeasured. Public integration andGPU acceptance pending; no full-goal completion claim.
2026-10-09: Public downstream calibration regression suite passed remotely: 260 passed,5 skipped (source b8d48e7). Three new orchestration checks preserve actual requested beam and frozen inference micro, reject real-beam admission failure, and mark finite-graph calibration unverified. Evidence published GitHub; GPU public-hook acceptance still pending. Full-frontier sweep PID4287 remains live, run10 timings about36.9seconds/depth.
2026-10-09: Expanded CUTLASS readout probe with distinct deterministic random signed FP16 operands, biases and ordered coefficients for1/3/17heads and scalar/unaligned/aligned outputs. Independent CPU dot-product and final ties-to-even key oracle uses exact dyadic inputs. Built remotely (commit2f69e4b); acceptance waits for live profile sweep PID4287 to avoid timing interference. Waiting SSH session is bounded900seconds; GPU acceptance not yet claimed.
2026-10-09: Added sampled GPU temperature/thermal and external power-brake event receipts (5second interval) to inference and downstream calibration. Observed throttling excludes samples from statistical selection; unavailable telemetry remains explicit errors, not a clean-hardware claim. Parser/orchestration tests10passed remotely at256e8b6. Inference launch failure now closes the failed log and cleans already-started ranks/monitor. Fullsuite after0287571 pending.
2026-10-09: Full Python suite after inference telemetry changes passed264tests,5skipped in15.19s at0287571. Random CPU-oracle CUDA probe compiled and queued behind live sweep; waiter session70967. Sweep confirmed PID4287 live at23:10, run14. No profile selection or publicGPU acceptance claimed.
2026-10-09: Fixed real public ensemble CUTLASS build: configure now enables LibTorch backbones and cache signature includes torch ABI/version for ensemble backend. Regression build tests13passed remotely at4e31e89. Prepared full publicAPI acceptance on lrx8 with three Hamming heads, actual2GPU auto inference+pipeline calibration, beam8192 and solution replay. Queued behind live sweep/readout probes; remote build only, session pending. Goal remains active.
2026-10-09: Bumped inference calibration signature toschema3; cache acceptance now requires winner telemetry receipt and rejects any observed throttling even if top-level clean flag disagrees. Added cache corruption/legacy rejection regression. PublicAPI build/probe remains queued behind PID4287; source preparation completed before build starts.
2026-10-09: Documented development ensemble API, ordered FP32 final-epilogue blend, inference-first/downstream calibration, bounded-frontier limits and128rank layout-only boundary. Strengthened pending publicGPU acceptance to require native_metadata inference_autotuned and pipeline_measured, not merely found/replayed solution. Inference cache schema3 regression3passed remotely (014ebf7).
2026-10-09: Full-frontier sweep completed7profiles; statistical winner baseline infer1024/outer1024, median35.056591microseconds/parent (~36.76s/global1048576depth),5postwarmupsamples. Selection archived c1d67b9. Independent random probe initially linked stale /workspace/ensemble-readout.o and rejected scalar outputs; rebuilding against current CUDA source. PublicAPI initial fixture selector cuda:0 unsupported by installed CayleyPy; corrected tocuda (54f414c), preserved failed-run directory and restarted terminalconfirmed driver. No GPU public acceptance yet.
2026-10-09: Independent random CPU oracle CUTLASS acceptance passed bothRTX3060,1/3/17heads,outputs1/3/24,257row tails,signed coefficient sequences and final quantization. Existing constant1/2/3/8/17head andNaNsticky guards also passed. Current-source standalone binary/source SHA receipt archived GitHub e4fc06e. PublicAPI v2 driver confirmed live PID6052, preparation/build in progress; session52362.
2026-10-09: Explicitly bind ensemble worker executor to LibTorch family launcher even if executor metadata evolves; protocol regression asserts BEAM_BLEND_DIR andlibtorch_eager. Backend suite54passed remotely (25ca59b). PublicAPI driver6052 confirmed live; runnerbuilt, calibrationhelper compiling. Source nativebytes unchanged duringbuild; Pythonbackendupdate does not alter already-loaded driver module.
2026-10-09: PublicAPI sourceboundrunner/helperbuilt; driver6052 reached real2GPU inference selection and full-depth downstream measurements on actualbeam8192. Native logs show4096parents/rank and6depthcompletion. Final publicsolutionreplay stillpending. Normalized separate Cube3head inference36.40918s vsfullpipeline36.7595s atglobal1M, not pairedoverhead claim; archived comparison GitHub.
2026-10-09: Public three-head ensemble GPU acceptance passed actual2RTX3060,lrx8,state8/moves3,beam8192. Bothranklogs confirm ensemble_model_count3,FP16execution/FP32accumulation; inference batch256frozen;19downstream profiles measured, selectedStream4batch16384/outer256. Foundlength4path and replayvalidated; pipeline measurement coversfullrequestedfrontier. Receipt archived ca29bf33; workerlogs archived separately. Single-model native-executor calibration remains pending; do not infer that goal complete from ensemble acceptance.
2026-10-09: Added actual native CUTLASS MLP Stream1 timing executable with independent entireLibTorch FP16/BF16 backbone oracle beforetiming. Graph-derived paddedstates/scalar-child expansion, runtimeweightview/scratch,3warmups+7event samples; no replacementofproductionnativeexecutor. Publisheda447b47 and remote helper-only compilation started; GPU acceptance/public single-model calibration wiring pending.
2026-10-09: NativeMLP helper buildpassed; independent backbone oracle initially foundexact3072keyoffset. Rootcause existing src/config.hpp STREAM1_SINGLE_OUTPUT_SCORE_OFFSET=3 added bynative scalar-child quantizer. Preservedproduction convention; oracle nowexplicitlyadds+3 and receipt exposesnative_scalar_score_offset. Currenthelperb5b6163 compiling; prior diagnostics preserved. No nativecode semantics changed orGPU calibration acceptance claimed yet.
2026-10-09: NativeMLP probe acceptance passed bothRTX3060,batches32and256,8192legalparents,7repeats, independentwholebackbone oracle exactkeyerror0 afterexplicit existing+3scalarscoreconvention. Nativekernel timing32batch~5.30ms/8192parents;256batch~0.79ms. HelperbinarySHA a1f04d6bd0424bd630ee52b87a9f4297b1cb1ca986cd97af8c70ea54f900bc15. Publicsinglemodel autotuning stillnotwired; do notpromote microprobe asAPIacceptance.
2026-10-09: Wired native CUTLASS singleMLP helper into build cache identity, binary validation/snapshot and public inference/downstream calibration. Preserve native executor and existing+3scalar score; convertparentbatch tochildrowbudget. Native downstream fixes actualStream1 batch while tuningrings/shards/sort; ensemble outertuning unchanged. Corrected candidate-count metadata aftercalibration. Added scalarrowbudget/freeze orchestration regressions; publishedc4b2dd9. Full Python gate andpublic nativeGPU acceptance pending.
2026-10-09: Full Python suite after native singleton calibration wiring passed269tests,5skipped in14.42s (c4b2dd9). Prepared/started fullpublicGPU singleMLP acceptance50b9999 on2RTX3060, lrx8,beam8192, fixednativeCUTLASS targetproduction_runner; assertnativechild rowbudget3*parentbatch andbothcalibration stages/replay. Runner maybuild shape/cachefresh; no successclaimed untilterminalreceipt.
2026-10-09: Native single-model public acceptance live PID9366/session6362 uses frozen b5b6163 native eager timing helper; do not update native source duringbuild. Identified remaining fidelity gap: productionnativeCUDAgraph vs helper eager launches, and unbounded oracle onehot memory. Locally implemented dedicated nonblocking stream/CUDAgraph replay and128parent oraclechunks; published separately, not deployedto livebuild yet. Must GPUvalidate graphhelper andrerun updated publicacceptance before completion.
2026-10-09: Started isolatedCUDAgraph probe build outside /workspace/source to preserve runningpublicnative sourceSHA gate. Compiler/link flags from existinglrx8target and bound native beam_cuda/beam_core archives; receipt records source/binary/archiveSHA andexactcommands. Nativegraphhelpera9228a7; buildscriptfb3aa62. Do not runGPUprobe concurrently withpublicnativecalibration PID9366.

2026-10-09: Fresh Vast inventory confirms only owned 54987544 and other-project 54998742; obsolete owned leases absent. Native MLP calibration admission failed due to blend-only benchmark guard, not VRAM. Published generic calibration guard/barrier fix b4ef6c12; CUDA Graph helper GPU0 batch256 passed whole-model oracle with key error 0. Started public native-single v2 acceptance with immutable updated source, pending build/GPU gates.

2026-10-09: Public native single-MLP CUDA Graph autotune PASS on both RTX3060: full requested beam8192, parent batch256/row budget768 frozen, downstream sort16384, solution length4 replay PASS. Receipt0b1c89e and raw cd6b3d1 GitHub. Added actual ordinary LibTorch MLP calibration, ATen reserve in native admission, and helper-name/digest validation; Python275 passed/5 skipped. Isolated LibTorch helper both3060 batch-invariance key error0; public LibTorch and one-selected-GPU acceptance started, pending.

2026-10-09: Deadline watchdog process absent (psutil verified); restarted exact-owned lease watcher session55555. README now distinguishes native CUDA Graph, LibTorch eager, ensemble calibration and conservative profiles for families without a dedicated probe.

2026-10-09: Serial native single selected GPU1 and forced LibTorch two-GPU public calibration/replay PASS with fresh inference profiles; receipt2baf516f GitHub includes raw logs. Published source initially omitted native build dependencies: CMake/quoted-include closure has176 files; created109 missing files and aligned15 existing source files at5bf57ba9. Archive download live session67202. Complete distribution manifest verification, corrected source pin and installed-wheel acceptance remain pending; local f377 source pin is provisional and must not be published.

2026-10-09: Created compact source-only tree/commit b2403731 and installation branch codex/ensemble-autotune-source-20261009 (5403dfb1 source-pin update), preserving old repo assets. Archive585041B SHA256d27f76bf...; all176 native closure files byte-verified. Built and installed matching wheel only on owned Vast host. Started clean installed-wheel 3-head/2GPU public acceptance without explicit source/CUTLASS paths, session42988, still pending. Cancelled only obsolete slow full-repository curl PID18258.

2026-10-09: Installed compact-wheel source/128-rank/per-candidate-timeout CPU gate27 passed. Fresh wheel GPU driver PID18414/session42988 live, automatically downloading pinned CUTLASS archive (8MiB observed at3min). Native build/search acceptance is pending; no restart on observation timeout.
2026-10-09: Inference selection now includes actual non-power cap and local refinement around coarse winner, schema4 cache isolation,70/30 coarse/refinement budget. Nine CPU regressions PASS including two-rank384 winner. Compact branch5d2ff3b5 has updated selector. Corrected installed-wheel harness to check source/CUTLASS digests instead of intentionally cleared prepared options; v2 fresh wheel GPU acceptance session83006 pending.
2026-10-09: Fresh installed wheel681868B SHA75af2e80 passed3/17-head2GPU full calibration/search and independently replayed paths4. Installed CPU suite285passed5skipped. Harness-only source-path/backbone-selector assertions corrected; terminal successful searches were recovered from persisted artifacts without repeated GPU runs. Content-addressed archive349879B SHA310bba33 contains1256 files, published112369017 and round-trip verified f1ddddfb. Owned lease54987544 stopped,destroyed,confirmed absent. Completion audit records128-GPU planning-only and registered-family limits.
2026-10-09: Source-audited fast calibration sizing. Added isolated size_model.py with actual R/Q/C integer geometry and named-allocation lower bound, exact-effective-frontier shortlist and linear service/latency equations. Calculations cover1/10/100/256M on8GPU at inherited gamma1.25/4. Prototype reproduces earlier1M Cube Q196608,R6,C524288. No GPU rental or production behavior change. Design identifies repeated process/model startup, capacity-sized Stream4 arena and beam-anchor cache identity as next implementation targets.

2026-10-09: Started fast autotune implementation. Inference cache schema5 keys exact requested beam instead of power-of-two anchor; added calibration wall time. GPU acceptance pending.

2026-10-09: Persistent per-GPU inference sessions and all-rank native memory planning; schema5 exact-beam cache; bounded service probes at actual C/Q, independent capacity search across1..128shards. Full-frontier test separated from fast component calibration; symmetric-orbit certificate permits unique legal factorial-rank fixtures. 2GPU preliminary public replay passes8192/65536/1048576/max89,530,368; max still memory/component admission only. 8GPU lease55039547 preparing. GPU acceptance pending for latest service-derived J/staging selection.

2026-10-09: 8GPU public replay passes65536/1048576/max334,397,440 using ready binary433d4758 transferred through checksummed GitHub multipart artifacta6a9a324. Fast calibration93–99s exposed process startup overhead; persistent exact-capacity service prepared locally/source10c80844, GPU gate pending. 2GPU90M full-depth test stalls afterexchange9157; original conservative A/B CDF cannot certify some pressure pruning. Added read-only exact A/B union histogram + native CUB pressure compaction preserving all global-selection semantics; independent CUDA oracle and full-pressure validation pending. No practical max completion claim.

### 2026-10-09 — persistent tuning / pressure acceptance
- Native component probes reuse one process per GPU and gate NCCL on all-rank allocation success.
- Added protocol regression tests for the allocation gate and cancellation.
- 2xRTX3060 exact A/B union histogram CUDA oracle passes through 65536 capacity, overlap/min-score/empty/partial-warp cases.
- A bounded full legal LRX13 frontier step at 89,530,368 completed on both ranks: 50.5732 / 50.5734 seconds. This is one-step correctness/progress evidence, not repeated timing acceptance.
- Source pin and 8GPU acceptance remain pending.
- Extended bounded component shortlist to outer dispatch batch and concurrent sort lanes, preserving frozen inference microbatch and exact frontier. Added contract test; Linux full suite 302 passed, 5 skipped on 8x3060 host before GPU timing.
- Automatic source setup pinned to immutable verified native revision d52f81ff with downloaded archive checksum.
- Full-frontier verification now compares a proxy winner with the baseline for effective beams <=1,048,576 and requires statistically separated actual improvement; regression protects against slower proxy winner.
- Latest user requirement: exact beam-dependent automatic inference-first parameter tuning; validate multiple beams and maximum on 2x3060/8x3060; total new-rental budget USD20 including preparation/storage/traffic. No subagents. Raw GPU evidence stays on GitHub/HF.
- Console distinguishes best microbatch calibration throughput from matched full-frontier Stream1 throughput; the latter is used for actual full-step loss.
- Found and fixed maximum mode retaining its bootstrap 1M inference receipt. Maximum now repeats inference calibration for its exact width, re-admits changed microbatch/reserve, and rejects mismatched cache identities or nonconvergence. Three regression cases added; 8GPU control-path acceptance pending.
- 2GPU results archived and all 947 files verified from GitHub; instance 55029041 stopped/destroyed and absence confirmed. Subsequent Python control-path changes do not alter native binary.
- Write component selection and completed full-frontier measurements before independent matched-inference measurement, so a later timeout does not discard completed evidence.
- Added large-beam throughput batch candidate alongside latency-derived J: amortize C-sized Stream4 merges without changing C, semantic selection, inference microbatch or exact frontier. Native admission still gates the proposal; GPU timing pending.
- Corrected component selection objective: max-only overlap envelope tied large-J candidates when NCCL dominated and ignored repeated sort pressure. Selection now minimizes summed measured service work; retains max envelope as diagnostic only. Sum is explicitly a scheduling proxy, not claimed wall time.

- Final default-batch 2x3060 acceptance (source ee50c804, native d52f81ff / runner 611ccc84): public path replay and full-frontier warmup + five measured depths + matched inference passed for 65,536, 1,048,576 and maximum 88,891,392; selected inference batch 8192. Maximum complete depth median 14.63s versus matched Stream1 6.52868s. Archive fb17b26d has 337 files, all SHA256 roundtrip verified through GitHub; owned instance 55053060 stopped, destroyed, absence confirmed. README corrected stale small-frontier-transfer description, development branch and bounded-search/optional-full-check semantics. 8GPU maximum default-batch acceptance remains running.

- Final 8x3060 default-batch acceptance: legal full-frontier warmup plus five complete depths and matched inference passed for 65,536, 1,048,576 and maximum 327,942,144. Maximum median full depth 104.751s versus Stream1 6.03996s; inference batch8192. User-requested small-beam audit completed with same-fixture profile sweep, clean controls and two Nsight CUDA/NVTX reports. Owned host has no supported P2P and two NUMA groups; 1M baseline submits all local parents in initial fill, so prefill cannot be assumed to create more hidden inference work. End-of-depth global materialization/selection tail observed; no causal general prefill/profile optimum claimed from varying communicator contexts. README documents overlap limits. Eight-GPU archive b3314be3 contains1938 files, every SHA verified from GitHub; instance55039547 stopped/destroyed, absence confirmed. Final2 archive337 files and original2 archive947 files verified; all three owned leases ended. No GPU archive copied to Windows, no unrelated lease touched.

2026-10-09: Prepared real Cube444 trained Transformer s3 + MLP mlp_x16 FP16 CUTLASS ensemble benchmark (not Hamming), immutable native source d52f81f, source-bound input hashes, numerical gate and bounded batch scan. Lease55065930, USD0.144444/h, three-hour stop guard. GPU execution pending image load. Public Python adapter currently MLP-only; native C++ Cube444 families exist.

2026-10-09: Real Cube444 FP16 Transformer s3 + MLP mlp_x16 blend GPU validation: isolated inference and 1M/8M full depths completed on 2x3060. Fixed calibration cache carry-over; confirmed runtime planner double-counted native weights/scratch under explicitly reserved LibTorch executor, accounting fix published c1917b6c99c0048bd1f309915f7e3cac677be3dd and rebuilt remotely. Exact-frontier re-admission and matched timing remain pending; no public CubeTransformer adapter or 8GPU neural acceptance claimed.

2026-10-09: Maximum-frontier bootstrap now chooses the minimum measured inference reserve rather than the fastest high-memory microbatch, then retains exact-width inference refinement/admission. Three new contract tests plus four admission regressions pass (7 tests). Source branch commit2ee4d4e1602b91502406f2c2f6333df4c70b3d68; native accounting pin c1917b6c99c0048bd1f309915f7e3cac677be3dd. Real 2x3060 FP16 blend full-depth acceptance is ongoing; 8x3060 currently has no matching rentable offer.
2026-10-09: Trained Cube444 blend final 2GPU matched acceptance: 1M Stream1 34.00355s/full34.4901s/loss1.4107%; 8M272.16247s/273.469s/loss0.4778%. GPU numerical/fixture/path/accounting gates PASS, final Linux314 passed5skipped. Evidence972d523b and proof00a9f82d roundtrip228files verified; lease55065930 destroyed and absence confirmed. New bounded benchmark repeats CLI validated on trained blend and archived635ffed2 at00a9f82d; built only on cheap pair. 8GPU trained1M measured loss14.9789%, baseline component profile; diagnosis and maximum validation pending.
2026-10-09: Documented maximum frontier versus inference throughput tradeoff: minimum verified inference footprint bootstraps capacity discovery, followed by exact-width recalibration and fastest admitted batch. Smallest reserve is not described as fastest inference.
2026-10-09: Bounded downstream candidate isolation fix: outer-double retains baseline flush batch and halves staging (minimum2) to limit producer-buffer growth; sort-lane candidate retains baseline flush/staging. Previously both silently inherited service-geometry changes, confounding capacity rejection and measured service effects. Regression expanded; real same-frontier neural 8GPU A/B pending.
2026-10-09: Published isolated profile candidate fix1630aadf2b4d09f189a9cbf2bbf4a3f16899af9e (six targeted regressions pass on Windows, exit0; pyreadline destructor emitted WinError afterward). 8GPU trained8M matched result67.17878s Stream1/69.667s full/loss3.5716%; real32M fixed-frontier timing still running. Owned8GPU stop guard extended to2026-10-10 06:00UTC within the existingUSD20 total for bounded single-pass maximum neural validation; conservative quoted total reserveUSD17.5, not a billing receipt. Native binary reused; no compilation on8GPU host.
\n2026-10-10: Calibration rank startup now launches all selected GPUs before waiting for readiness (two lifecycle regressions pass). Added CPU admission for existing Cube444 s3/mlp_x16 lossless bundles: full tensor schemas, hashes, FP16 representability, encoding indices and graph shape checked; execution precision distinguished from FP32 source. Ensemble members sharing an artifact reuse the same native cache path. Actual public trained-model calibration acceptance remains pending; no speedup claimed from source tests.
2026-10-10: CPU admission of the actual trained Cube444 bundle passed remotely (124 files, FP32 sources/FP16 execution, shared transformer/MLP hash, 1.62033s); 14 local artifact/session/ensemble regressions pass. Source published51ae75c40b81160bc2bd3171380665d19deac2c0. At21:45UTC SSH55084167 became unavailable; two API reads show owned record null and account inventory empty. Cause unknown, no destroy issued in this continuation. Partial1M/8M evidence survives at a80beb35; final32M/maximum archive is not verified and cannot be claimed saved.
2026-10-10: Strict two-observation selected-GPU telemetry checks added to inference selection, cache and fallback admission. Actual public trained calibration exposed PyTorch bare UUID versus NVML GPU-prefixed UUID mismatch; normalized only that prefix/case and preserved per-candidate receipts before statistical selection. Source27bbcc69fcb37020e64298db2a8fe96dc66fea9f. Owned recovery lease55101031 atUSD1.07335/h reuses remotely precompiled runtime; stop guard05:30UTC, conservative prior spend boundUSD8.5, total limitUSD20. Other current BFS lease explicitly untouched.

2026-10-10: Actual trained 8x3060 calibration exposed a noisy baseline abort despite complete stable alternatives. Added inference-only stable reference fallback; correctness, cohort, telemetry and timing stability remain required. Cache schema 8/policy upper-first-stable-reference-v3. GPU acceptance pending. Dedicated 2x3060 55105798 running within existing USD20 budget, deadline 05:15 UTC, watch 99225; unrelated BFS lease untouched.

2026-10-10: Inference stable-reference fix 38ea788ba973412fbfa1e1b0db42b1890b242e20 passed 23 CPU regressions. Recovery8 actual public 1M/8M calibration now uses exactly eight process starts; 8M cold85.02s/cache2.75s, batch8192. Pair actual public calibration frozen directly remote-to-GitHub commit4a08895939b6a23c2d5d66e507042759297c930b, 63379bytes SHA4dbeaa51aac5be5fc4bf6bfa19db76069814d07bbb8b2393872f589bc6f50405, 53files round-trip proof9c4da948a2b59639047c392dd2250c390b39b0d6. Harness counted duplicate next_frontier_size logger fields; exact calibration rows parsing repaired, completed six native depths reused rather than rerun. Original pair return codes were not persisted before harness assertion; this limitation is explicit in recovered exit receipt.

2026-10-10: Recovery8 source38ea actual public 1M/8M/32M calibration PASS. 32M preserved requested frontier, rejected fastest8192 reserve under native memory admission and selected measured2048. Calibration archive3998bc00b42946a4a4aa4425c6a63df27dd096ed contains233files/213957bytes SHA b58de4712bf7b6037e0cded5aad560352ed9e6d09f6c6b904fdc52766a58ba3c; remote round-trip SHA verified and proofpublished b43da663919ffe03b27bd73a71fb377d594576c6. Dedicatedpair 1M matched Stream1 32.550458984375s vs complete step32.9924s, throughputloss1.3395%; six completed native depths/counts/padding/frontier verified, originalreturncode caveat retained. Full8M pair and 1M/8M/32M recovery8, maximum tests and serialized A/B/NVTX diagnostics active or queued; archives_verified remains false for both leases.

2026-10-10: Completed trained-blend depths frozen remotely and published15f4f3093f069479e964072f9a47b8b3293e6fd7: pair1M90files SHA9d684b849289b66cd6095ca72b3cc55821b2fa4d9b333f8cb1d470cc61e5ca24; recovery8 1M/8M369files SHA6ba8c62dce2174bc2a63012f4af595d1046858db97c93984a04081e526b70a46. Both complete roundtrip verifications PASS. Recovery8 matched1M loss14.7811%,8M loss3.10049%; larger and maximum acceptance pending. Added exact-workload component-profile cache with frozen inference signature/reserve and mandatory fresh all-rank memory/geometry admission. No full-depth certification is inherited from component receipts; changed cohort/width/environment/geometry rejects reuse. CPU and actual public cold/warm acceptance ongoing.

2026-10-10: Exact-width component cache9 CPU regressions PASS41.11s. Source573a82aed2518dc2d729e961f0359bbbf90abdf3, tree5aa576a222c1d780270d82f451481fcb4fe90cc8. Actual GPU full-step source remains38ea(8GPU)/4ca(2GPU), cache acceptance queued strict public CayleyPy cold/warm after maximum, CPU full suite queued latest573. Partial completed-full evidence roundtrip proofpublished fb5fd01f50bbb9be7b5b64498872b446175ef088. No whole-lease archive flag until all remaining results verified.

2026-10-10: Public cold/warm API acceptance on latest573 scheduled serially: pair after fixed full1M/8M, before maximum; recovery8 after maximum A/B traces and Linux suite. Warm call requires both inference and component caches hit, native backend and independent solution replay.8GPU input frontier65536,2GPU8192, exact cache identities differ. Diagnostic SQLite analyzer queued read-only after traces; one-rank kernel work is descriptive and not complete-step acceptance.

2026-10-10: Recovery8 32M native completed six full depths on all8ranks, exact4,194,304 parents/rank and next frontier, codesall0. Matched isolated inference still running. Frozen terminal-native evidence406files361551bytesSHA09b2313adc3cba4c1edfa9ddf4ecbf453c3e5c07a463b01b4957cfee59472298 published496ca2254dc4f759124fdcbd2a9b7aa4664d6689. No measured full-step loss asserted until matched phase completes. Fixture audit: each parent is an independent80-move legal walk, not80repeatedstates; all colors/paddingchecked and16CPUreplays/rank, no exhaustive uniqueness assertion.

2026-10-10: Strict public CayleyPy API source573 actualpair PASS3move replay; trainedartifactc435..., FP16/CUTLASSheads. Coldcall64.984s/autotune52.813s; warmcall20.449s/autotune8.648s, both inference and component caches hit. Actualpair8M full261.9s vs matchedS1260.446625s, throughputloss0.554935%; allsixdepths codes0 exactfrontier. Archive205files308993bytesSHA4a5ad8d3ea6ba093c96b6a29ac434cf23efdd5a677ca582ed44725a894846093 published4b79fc4417ce0275d56f1a1d9870dd809b33d044. Maximum harness originally used absent p002 central_state key; fixed both capacity phases to pinned solved_state.npy before GPU maximum work, failedprelude retained under maximum-preamble-failed-v1. CurrentpairmaximumPID2350; recovery8 waitingmaximum10503 replaces5782. Native capacity bootstrappair55,130,112 states under measuredfloor32, searchcomplete1425probes1.681s, actualfullmaximumstillpending.

2026-10-10: Native32M archive roundtrip406files proof8f8ecf39586aa0a5f544d64fdee93436dfe6fd57. PublicAPIpair archive roundtrip205files proof67f5959ad488e623a2e7752dfb1538faefbc3af2. Queued diagnostic A/B extends existing sort/NCCL candidates with at mosttwo previouslyverified microbatches within2% isolated performance plateau, fresh native geometry, to test whether smaller workspace lowers the complete1M8GPU tail. Production frozen-batch policy unchanged until actual A/B evidence; waiting helper10544. Budget/currentleaseownership guards unchanged.

2026-10-10: Maximum harness stale eligibility caused a32/64reserve cycle: standalonefresh32peak186646528, fresh64peak180355072, while oldpool64peak182452224 incorrectly rejected current55,050,240capacity. Eligibility proposals now replace each batch's own reserve with its latest verified receipt; final throughput selection remains freshly measured at exact beam. Both control scripts patched, pairrestart3420 and waiting recovery8 helper10570. Pairconverged three rounds to55,050,240states/batch64 and fixture generation began. Queued timeline helper10613 now traces both baseline and selected isolated batches if different, preventing mismatched-shape contention claims. No five-stream native change.

## 2026-10-10 — exact maximum startup CUDA Graph pool
- Full 55,050,240-state 2×3060 acceptance exposed CUDA Graph instantiate OOM with 5,974 physical Stream3 rings; admission had fitted static bytes but not graph executables.
- Existing dispatcher returns rings to Free after Stream3/Stream5 and advances original frontier_cursor until every parent is consumed. A fixed 16-ring manual diagnostic passed graph creation and began the same exact frontier; completion pending.
- Added explicit BEAM_RING_COUNT_LIMIT to derived automatic ring pools, validated positive uint32 before allocation. This bounds physical buffers/graph instances without changing beam, semantic shard capacity, candidate identity, or five-stream architecture. Default unchanged pending measured policy acceptance.
- Component cache now requires fresh physical RING_COUNT equality; stale old receipts without the field miss safely. Added regression tests for cap validation/frontier preservation/cache geometry.

- Native eager ensemble default physical ring limit is16 (explicit BEAM_RING_COUNT_LIMIT override); native-graph executor default preserved. This is a bounded scheduling baseline, not a claim that16 is globally optimal. GPU automatic acceptance still pending.

## 2026-10-10 — downstream selection with frozen expensive inference
- GPU8 1M A/B: full baseline9.64022s; near-plateau2048 fresh geometry9.36155s; staging-only2 with unchanged NN8192/J262144/shards8/sort4 gave9.53467s. Independent staging alone did not beat the2% material-gain gate; no cause claim about cache from these timings.
- Source criterion included constant Stream1 in its5% gain denominator. Heavy real NN could therefore hide meaningful downstream improvements. Component selector now uses summed downstream service work (Stream3/Stream4/transport/union), keeps frozen inference, records total service separately, remains an explicitly unverified wall-time proxy.
- Added independent staging-only proposal while preserving sort flush; bounded at six alternative exact-capacity candidates. Component cache policy/schema2 invalidates prior-policy receipts.
- Regression compares identical downstream probes under cheap and extremely expensive fixed inference; winner must not change merely because the constant inference term grows. CPU/GPU verification status follows in test_results.

## 2026-10-10 bounded physical-ring shortlist
Python memory pruning now applies the eager native physical-ring pool bound (16 by default; explicit positive uint32 override supported). Logical frontier, owner capacity, and sort geometry are preserved. This is only a lower-bound shortlist; native all-rank admission remains mandatory, including for older prepared runtimes. Component-cache schema bumped to 3 to invalidate receipts selected using the unbounded-ring pruning policy. Causal ring 1→2 GPU A/B on trained FP16 blend improved 1M full depth by 2.81% (8192 microbatch/staging2) and 1.28% (2048/staging8), so the single-ring explanation is only partial.

2026-10-10: Document bounded eager physical ring pools in the primary architecture contract without changing logical frontier or five-stream ownership; clarify downstream-only selection objective and registered Cube444 artifact adapters. GPU ring4/8 causal overlap checks remain diagnostics pending results.

2026-10-10: Added test_results/real_neural_autotune_acceptance_20261010.md evidence ledger retaining native automatic-cap/public API/max-frontier/teardown gates. Ring4→8 causal A/B showed9.16141→9.15333s at fixed1M/2048; no universal ring floor introduced. Raw supplement98files roundtrip verified on GitHub.

2026-10-10: Clarified that the eager16 physical-pool default applies to automatic derived geometry; manual explicit BEAM_RING_COUNT remains a separately admitted request.

2026-10-10: Manual55M exact full2303.43s/matchedS12266.9845s (1.5822% throughput loss) completed with numeric oracles; archived17files roundtripPASS b994e8adaff52932a88d4e5886a3ed558e406d0c. 8GPU provider brief offline recovered with assets intact; public controller restart fixed empty-output-directory race and preserves prior control failure; no timed run repeated.

2026-10-10: Cheap native SM86 bounded-pool build passed runtime arithmetic tests; prepared archive1b25a1f verified on2GPU. 8GPU provider proxy failure handled via observed direct endpoint and background GitHub downloader; same machine assets/processes retained, no new rental.

## 2026-10-10 Direct local native launcher candidate
- Requirement: fast automatic profiling on trained Transformer+MLP, preserving five-stream semantics.
- Replaced single-host torchrun wrapper with simultaneous direct native rank processes, file NCCL rendezvous, rank-specific logs and bounded cohort teardown.
- GPU validation and speed acceptance pending; original public 8GPU warm total 101.427s, native elapsed 56.923s, actual search 5.947s. Exact-max 8GPU tiny-batch timing unstable; no acceptance claim.

2026-10-10: direct native launcher real8GPU PASS, but no speed acceptance (108.945s total vs prior101.427s); two fresh planner groups10.759/10.311s measured. Added explicit retained admitted planner ownership for fixed beams, preserving fresh exact-profile admission, closing rejected candidates and verification failures, downstream deadline reset; CPU62PASS2skip. GPU reuse validation pending. Full local suite unavailable: installed CayleyPy lacks ModelConfig, remote pinned suite required.

2026-10-10: retained planner8GPU real public PASS: total94.279s/cal23.145s/native8.469s vs direct baseline108.945/cal35.852/native6.392. Single observations, no statistical wall-time speed claim. Added shared Cube bundle snapshot/validation dedupe with explicit per-head expected hash checks, CPU14PASS; GPU acceptance pending. 8GPU largest exact stable admission211927040/B32; original protected-time precheck aborted before full, resumed unchanged preparation30624 under updated owned stop06:45UTC. Total conservative18.87+0.75<. Native full+matched acceptance pending.

2026-10-10: Inference calibration now distinguishes explicit CUDA allocation failures from numerical/protocol/timeouts. A failed initial batch falls back to smaller measured batches without changing requested frontier, including after coarse-phase expiry. Planner startup reserve rejection maps to CapacityRejected and closes all owners. Capacity-limited caches capture selected UUID free memory and are invalidated when headroom increases by more than 32 MiB; ordinary caches retain fast reuse. CPU acceptance: test_capacity_fallback plus calibration_session, plan_session, autotune_cache. GPU memory-pressure acceptance pending. Automatic 2x3060 maximum 55,050,240 matched full depth PASS: full2307.36s/pure2264.719s, throughput loss1.848%; one descriptive point only. Latest public API attempt failed before GPU because pytest was absent; dependency installed, bounded retry required.
2026-10-10: Strict max API exposed inherited BEAM_ENSEMBLE_INFERENCE_MICRO=64 sizing calibration workspace below next256 batch. Probe environment now copies caller env and removes only that frozen override; explicit helper capacity governs calibration arena, caller remains unchanged. Preserved failed GPU logs; strict API retry pending.
2026-10-10: Real trained FP16 blend pressure acceptance PASS on2x3060. External9GiB/card caused8192/4096 CUDA OOM;1M frontier preserved, admitted512 and three-move replay passed (98.883s). Releasing external allocations invalidated pressure-limited cache, selected8192 and replay passed (82.343s). Strict max API b9f5b40 PASS53,790,720 cold183.311s/warm40.636s; full maximum acceptance separate55,050,240 matched PASS. Initial maximum-capacity bootstrap now measures one stable small cohort (<=32), using the same explicit probe workspace capacity as subsequent exact-width inference to avoid reserve-induced width churn. Full performance sweep remains bound to discovered frontier; separate cache key. Local65 passed2 skipped16.61s. Fast-bootstrap GPU acceptance pending.
2026-10-10: GPU fast seed acceptance sourcea558fada PASS max53,790,720. Cold130.680s vs prior183.311s (single comparison~29% reduction), warm40.499s; no universal latency/optimum claim. Terminal924-file1.466MB GH archive f3debeb1 with independent roundtrip proof4fb5f2e;2GPU55105798 destroyed after verification, provider absent.8GPU maximum full+matched still live;20USD deadline unchanged.
2026-10-10: Documented proven allocation fallback, pressure-limited cache invalidation and measured <=32-row bounded maximum bootstrap. Documentation explicitly retains exact-frontier final calibration, fresh native admission, finite-search and full-depth evidence boundaries. No architecture changes.
2026-10-10: Own 8GPU maximum full step exceeds original 1.75x preliminary inference estimate. Remote full-window-guardian.py PID33717 verifies controller30624 command/start identity and all eight owned child PIDs, suspends only the Python controller while unchanged native workers continue, resumes on all terminal/nonzero exit or existing protected deadline06:35UTC minus30s. Lease stop06:45UTC and20USD limit unchanged. No full-step PASS yet; terminal archive fallback prepared and refuses snapshots while controller/owned GPU workers live. Guardian script and receipt included by terminal archive sources.
2026-10-10: User clarified inference batch takes priority over maximum frontier. Maximum capacity seed now chooses fastest verified cohort rather than minimum reserve; initial max calibration performs normal throughput sweep; exact-width refinement preserves its inference winner and resizes capacity instead of fixed-width admission downgrading batch. CPU validation pending; prior8GPU B32 full point remains capacity-first historical evidence, not new-policy acceptance.

2026-10-10 inference-first maximum policy: 18 CPU tests passed in 23.91s; source commit 70d65b437537f65bc05991b9d436b0e482cf80a4. GPU validation pending. Owned instance 55101031 stopped/destroyed after verified terminal GitHub archive f66c656a04a2116871f9bdeb789fac809bb45d46. Old full maximum PASS; matched comparison PARTIAL.

2026-10-10: Removed provisional 1M fixed-frontier admission before inference-first maximum sizing; its rejection could prevent shrinking the frontier for the fastest batch. GPU acceptance on own lease55160534, capped90min at0.1044444444/h, prior upper19.62USD incl reserve, total remains below20USD.

2026-10-10: Own failed CDI lease55160534 archived and deleted; replacement55160866,0.1087407407USD/h,60min hard stop1791617798,prior conservative upper19.80 including failed lease and traffic reserve. Controller417 live on /workspace/first2, remote prepared runtime download, real FP16 s3/mlp_x16 inference-first maximum public API plus one exact full step and matched pure pass planned; acceptance pending. Source0f57aff8dea3220135cf77c9cd7594b1f9581c9a; 60CPU passed2skipped.

2026-10-10: Real GPU proof found NVML thermal slowdown at170W; previous fastest8192 excluded then256 selected caused capacity oscillation. Added fail-fast thermal/missing telemetry winner guard so invalid evidence cannot expand maximum frontier. Own power limit change denied by provider; failed run archived remotely and GitHub.

2026-10-10: Thermal guard 19CPU tests passed; source5d65147dae7ec18a6bc2d58772f4f24a49da15ca. Thermal failed host55160866 archived c9b13d9905c6febbdcd0712c474ac3f2e50ee4d0, roundtrip92files passed, deleted and absent. Stable verification replacement55162682:0.0777777778USD/h,deadline1791618799,prior conservative upper19.70 incl traffic reserve and both failed hosts billed conservatively through06:52:22; authorized total20 unchanged.

2026-10-10: Pending host55162682 cancelled before any job, archived startup receipt and absent verified. Own replacement55163489 running;0.128888889USD/h,deadline1791619231,prior conservative19.78USD upper. Deployed latest5d65147 source acceptance, exact one-full-frontier step, bounded matched pure pass, remote-only archive controller; native runtime remains precompiled04f42d. GPU acceptance pending.

2026-10-10 final inference-first acceptance: trained s3+mlp_x16 FP16/CUTLASS,8192 model parents on2/8RTX3060;2 full5,533,696/172.775s matched171.043171875s;8 full1,703,936/13.6402s matched13.3145517578125s.8 fresh plan-only1,941,504 used268435456 additional budget bytes; different free snapshots, not a full-step certificate. Documented snapshot dependence. Archives independently byte-verified via GitHub; own2/8leases destroyed/provider absent. Source GPU acceptance5d65147. Main/Kaggle/128GPU not validated or released by this goal.


2026-10-10: Publish source fe31e073 as main, include independently verified evidence, update simple Kaggle notebook to GitHub install, preserve old branch heads before branch consolidation. Notebook syntax passed; Kaggle version4 GPU acceptance pending.

2026-10-10: Kaggle v4 exposed scalar MLP/Hamming row-budget versus parent-batch receipt mismatch. Validate outer candidates against admitted native parent batch and staging slots; no C++ changes. 14 focused CPU tests passed. Kaggle repeat pending.

2026-10-10: GitHub: exactly main remains; all26original heads verified retained at archive/20261010/*; consolidation run38042607200 succeeded. Main contains checked source and all selected reproducibility evidence. Kaggle public version5 installs source00a3729bc6d7e1743c11d21e68fac06c57216ce5 with GPU enabled and unchanged simple imports. Published contents independently pulled and verified. Version4 exposed scalar-MLP scorer-row-budget versus native-parent-batch mismatch; fixed in library,14 CPU tests passed. Version5 saved with official QUICK_SAVE execution type because batch GPU quota2 is occupied. No successful version5 GPU run is claimed. No extra Vast rental or unrelated notebook stopped.

## 2026-10-10 GNN candidate: GPU acceptance pending
Own dual-stream GATv2 and multi-hop pancake GNN, FP16 TorchScript export, immutable graph-bound snapshots, arbitrary-count ensemble integration, native LibTorch/CUTLASS projection paths, explicit backend selection and backend-specific calibration cache. CPU reference comparison and 36 focused tests passed. Linux/CUDA compilation and GPU acceptance are still pending; no trained-quality or speed claim.


## 2026-10-10 — own pancake GNN GPU acceptance
Own dual-stream GATv2 plus multi-hop neighbor inference added to native ensembles with LibTorch and aligned FP16 CUTLASS projections. T4 LibTorch and RTX3060 both-backend feature parity passed. RTX3060 full single/three-GNN beam paths replay correctly on both backends; blend readout oracle key error zero. See test_results/issue5_review_20261010/REPORT.md for scope, source commits, artifacts and untested training/performance/multi-GPU boundaries.


## 2026-10-10 — large GNN inference comparison
RTX3060 n100/d256/layers2/hops2 matched FP16 inference: batch3 native LibTorch549.54ms, CUTLASS671.27ms, Python700.55ms. Fitting batches1/2/3 pass parity, batch4 OOM for all. Weights seeded synthetic, no trained quality proof; common FP32 scalar readout, no full beam timing. Evidence and fingerprints in test_results/gnn_large_20261010/REPORT.md.


## 2026-10-10: native GNN CUTLASS fusion
Fused GAT aggregation and compact edge projection preserve both GNN levels and all beam-stream contracts. RTX3060 n100/d256 batch3: 652.268 -> 193.153 ms (3.377x); LibTorch 540.440 ms. 28 GPU parity cases, zero sanitizer errors, single/three-GNN full-beam replay pass. See test_results/gnn_opt_20261010/REPORT.md.


## 2026-10-10: GNN fused rounded bias
Removed separate bias launch using broadcast C and a double-rounded FP16 epilogue. RTX3060 full GNN AB/BA: 213.548 -> 178.821 ms (1.1942x). 28 projection cases and full native model outputs bitwise match; memcheck zero errors. See test_results/gnn_bias_20261010/REPORT.md.


## 2026-10-10: further GNN optimization with Astra
Accepted fused residual/LayerNorm/exact-GELU: RTX3060 matched full forward179.027 ->164.364ms, 8.92% throughput improvement. 120 GPU norm cases,9 CPU tests, zero sanitizer errors; final single/three-GNN production replay pass. Rejected parallel-softmax and wider-tile experiments. See test_results/gnn_further_20261010/REPORT.md.

2026-10-11: Retain compatibility commit 1ee313325a39371492c043a380f0af78cd7713ff as a merge parent while preserving current main source. Only legacy publisher state changes from published to accepted after HTTP 200/202. Three isolated transport-state tests pass (202, duplicate 200, failure 400). Cube555 notebook pins this solver-compatible commit; GitHub-only input metadata retained. No GPU run is claimed.

## 2026-10-25: Cube555 and hardware-adaptive profiles
Published graph/model/API integration, exact logical beam and hardware-derived coarse-to-fine calibration. Preserved existing beam architecture and frozen inference microbatch. Real single-GPU acceptance, replay and fresh cache reuse passed; see docs/validation/cube555-adaptive-20261025.md for exact workloads and limits.

## 2026-10-25 - Preserve a measured baseline after challenger failure
- An optional coarse-profile timeout no longer discards a verified baseline. Baseline failures still abort.
- Incomplete search is reported and excluded from complete-profile caching; only completed timing cohorts are selected.
- Validation: 18 pipeline/autotune cache regression tests passed. Clean Kaggle 2xT4 execution remains in progress.


## 2026-10-26 Service-guided profile shortlist and clean Kaggle validation
Measured-pressure shortlist reduces downstream tuning work while preserving inference batch and exact beam. Same-host Quadro RTX5000 paired checks at65536/262144 retained baseline full-step speed with about2.4-2.6x shorter tuning; replay and real cache validation passed. New trace distinguishes analytical jobs from CUDA kernels. Separate public notebook160 cells passed from scratch on2xT4: index1020 solved116moves, exact1048576beam, independent replay. See docs/validation/profile-shortlist-20261026.md and cube555-adaptive-20261025.md for evidence and limits.

2026-10-26: NativeSession retains compatible GPU workers, model weights and buffers across calls. Compatible profiles are reused. One RTX3060 and clean two-T4 simple/real FP16 blend acceptance passed, including changed starts, not-found recovery, beam invalidation and teardown. Corrected controller-only solved terminal handling. Both public notebook examples use explicit sessions and pinned source7c38e298. See docs/validation/persistent-search-20261026.md for scope and evidence.

## 2026-10-27 — Retry noisy full-capacity component calibration
When Stream 3/4 component timing makes the baseline cohort exceed the relative-MAD guard, preserve the requested beam and retry that baseline cohort once. A recovered retry is recorded in calibration output; a second unstable cohort or exhausted budget fails explicitly. Added regression coverage. Focused integration validation: 23 tests passed. Kaggle 2×T4 long-puzzle rerun is pending; no full-step or long-puzzle success is claimed yet.

The public Cube555 notebook now fetches the pinned setup script that installs the retry-capable source commit; the example's editable search settings are unchanged.

## 2026-10-31 — Cube555 2xT4 profile grid
- Added seven source-matched 2xTesla T4 profile records for global beams 65,536 through 4,194,304. Native source digest c00b842d2baedd7028f2c736af7a07c297b2cad7b652dea000ed0c625a59cad1 is reproduced from solver commit 804efdc28d84f1d547d5f0e7fe47b2c81ec3b928 after the pinned wheel's CRLF normalization and source allowlist.
- All 72 archived Kaggle output files matched the SHA-256 manifest; independent rank checks confirm one saturated full step per rank at each width. Times are single-step measurements, not medians or global-optimum claims. Stream1 slowdown is a derived estimate, not a second full-step benchmark. No solution was found.
- The previous 1,048,576 T4 entry remains unchanged under its legacy source/runtime key. The new profiles are exact-match proposals and still require fresh GPU admission, correctness, and timing validation.
- Public profile evidence summary: profiles/evidence/cube555-t4-grid-20261031.md. Detailed raw logs remain in the local test_results/kaggle_t4_profile_grid_20261030 archive.

## 2026-10-31 — Cube555 2xT4 profiles published and Kaggle v179 Quick Save
- Published seven exact-source 2xT4 profile records for beams 65,536–4,194,304 in commit a5dbf0bc; the README and evidence summary describe one full-step timing per width and derived Stream1 estimates.
- Pinned the public setup script to the profile-bearing source commit and updated the repository notebook in commits 92205f1e and aed7e4aa.
- Saved public Kaggle notebook version 179 with Kaggle CLI 2.2.5 `--no-run`. Pulled v179 back and confirmed the seven notebook cell sources match; no cells or GPU work ran during this save.
- Raw grid artifacts and measurement limitations remain under test_results/kaggle_t4_profile_grid_20261030/; save receipt and report are under test_results/kaggle_quicksave_20261031/.
- The notebook still imports the native adapter and Cube555 helper; the requested CayleyPy-only public API is a separate unfinished integration.


## 2026-10-31 — Generic CayleyPy graph and model API
- Kept `import multigpubeamsearch` as the public native execution package and removed the puzzle-specific `multigpubeamsearch.cube555` helper.
- The notebook can use a caller-supplied CayleyPy graph, supported predictor/model, and start state. Search, collect, BFS, reflection, replay validation, saving, and publication use the shared API; Cube555 remains the example model.
- Focused local checks: 10 passed. Public source and Kaggle notebook save are being verified separately; no GPU search was launched for this change.


## 2026-10-31 — Generic CayleyPy notebook verification complete
- Published source changes on main at `5f00f003d87374ee704f3629aef770ae8f20e420`; the generic API commit removes the Cube555 package helper and preserves existing profile records.
- Saved Kaggle notebook v180 with CLI 2.2.5 `--no-run`. Pull-back verified all seven cell sources and valid code-cell syntax; version-specific status is COMPLETE. No GPU execution occurred.
- Ten focused local tests passed. Native support remains limited to permutation graphs up to 150 entries and supported model exporters; custom graph/model GPU acceptance was not run.


## 2026-10-31 — Simplify Cube555 notebook
- Replaced the generic graph/predictor configuration branches with a single Cube555 input, graph, state and Transformer/MLP flow.
- Kept the first-cell controls for puzzle range, model weights, beam, collect, BFS, reflection, GPU count, output and publication.
- Preserved the generic `multigpubeamsearch` library API. Notebook publication to Kaggle is pending; no GPU search was started.


## 2026-10-31 — Cube555 Kaggle v181 Quick Save
- Published the four-cell Cube555 notebook on main at `64f4270c3eafd38f9348c315fd8bf29063289fd1`; its setup pins source `534f63516552081e48f5c575c6471a888c9db5b9`.
- Saved Kaggle v181 with CLI 2.2.5 `--no-run`. Pull-back confirmed all four cell sources match GitHub, code cells parse, and version-specific status is COMPLETE.
- No GPU search or pytest suite was run. The first visible cell retains the model/blend, beam, collect, BFS, reflection, GPU, output and publication settings.

## 2026-11-06 — issue #5 trained Pancake-10 comparison
- Added the alternate Level-1 Transformer to PancakeGNN with the same Level-2 GNN, trained weights and graph-bound native exporter. Restored GAT attention dropout and differentiable eager training projections.
- Trained Pilgrim, GATv2 and Transformer with the upstream Pilgrim protocol at a shared reduced budget on one RTX 3070. Across three initializations and 100 state-disjoint starts, solved counts were 300, 300 and 294. Optimal counts were 219, 268 and 107.
- Checkpoints, exact distances, independent path replay, source parity and reproduction scripts are in `test_results/issue5_training_20261106/`.


## 2026-11-06 — packed CNOT search and external multi-node ranks

- Added a packed GF(2) domain, CUDA row XOR, exact full-matrix CUB deduplication, native Stream 5 exchange and a distributed raw-value top B. History preserves the physical 16-bit gate ID through independent replay.
- Added an opt-in binding for the consumer's pinned 102-feature trained scorer, an independent small-instance oracle and runnable torchrun/Slurm examples.
- Split node-local CUDA assignment and RAM budgets from communicator-wide rank and width. The permutation launcher admits one shared geometry and preserves the requested global width on uneven partitions. Verification is recorded in `test_results/issue6_cnot/`.
- Verified the trained 102-feature checkpoint on two physical T4s in both head modes. A depth-three fixture expanded one parent on each GPU, kept the global two of four distinct children and replayed to the input at the exact BFS distance. A separate 16-process Gloo test selected the same 17 of 313 states as two GPUs; the physical multi-host run remains open.

## 2026-11-07 — recoverable repository consolidation
Preserved 26 checkouts and their source/index overlays; verified restoration of 24,134 files and all original Git heads. Repaired shallow-clone recovery using published-main ancestors. Archived old checkout directories without deleting their data, retained historical branch tags, consolidated the root to main 0eafb193 and moved build caches to .local on D. Existing inference and search implementations are unchanged at this stage. Rust/CUDA implementation and GPU acceptance remain in progress.


## 2026-11-07 — Rust/native context and persistent rank supervision
- Added beam-core/runtime/native-sys/CLI/PyO3 workspace, versioned coarse C ABI, lazy CUDA library loading and persistent rank workers with cancellation and group cleanup. Native contexts keep their library alive; borrowed input views are copied before submit returns. Specialized inference kernels and public launch defaults remain unchanged.
- Added native resource unwind cleanup and explicit history slot teardown ordering. Build caches and source distributions now include native/Rust sources and ABI inputs. Documented memory ownership and large copies.
- Verified remote builds on 2xRTX3060: native CTest57 pass/1 skip, Python510 pass/25 skip, Rust8 CPU tests plus real FFI/PyO3 checks. Native and LibTorch one/two-GPU repeated jobs match reference paths; cancellation and one-rank failure reap the whole group. Narrow memcheck passes; unfiltered NCCL API diagnostics remain recorded. See test_results/refactor_20261107/STAGE2.md.
- Fixed a baseline CPU test's FP32 operation-order assumption using an independent FP64 roundoff oracle while retaining exact export checks. No numerical inference policy change.
- History/profile/CNOT control transfer, CUDA optimization, public switching and full hardware/performance acceptance remain in progress.


## 2026-11-07 - Rust host history and calibration process ownership
- Moved host history planning/storage/pruning into beam-core/runtime. The optional ABI-1 callback table borrows ready pinned CandidateMeta batches without an extra record copy; C++ retains CUDA events and pinned-slot reader lifetimes. Rust contains callback panics and owns RAM/disk resources, including disk-failure fallback after all future RAM reservations.
- Extracted native history transport from production_runner into tools/history/candidate_history.hpp. Added the same provider to standalone Rust CLI calls. Reference C++ history remains until the public-switch gate.
- Added Rust ProbeSession/ProbeCohort and thin PyO3 bindings for retained calibration process groups, bounded JSON/log transport, deadlines and whole-cohort cleanup. Public profile policy and launch defaults are not switched yet.
- RTX3060 one/two-GPU history comparisons passed in RAM/disk/static-hybrid modes and with LibTorch. Cancellation and exact one-rank failure passed. Memory-only Compute Sanitizer with full leak checking reported zero errors and zero leaked bytes; API reporting was disabled for the separately recorded, explicitly handled NCCL kernel-query diagnostics.
- Python regression suite: 510 passed,25 skipped. Rust CPU/FFI/probe tests and Python adapter passed. Paired performance, CNOT control transfer and other GPU-family acceptance remain outstanding.


## 2026-11-07 - Rust exact planning and complete-step calibration control
- Added typed all-rank planning contracts, exact logical quotas without integer overflow, retained prepare-all/run-all component protocol, frozen inference microbatch and explicit-capacity-only bounded search.
- Added Rust-owned complete-step calibration processes, telemetry, warmup exclusion, conservative shortlist statistics and a separate paired optimization-promotion gate. The bootstrap RNG is explicitly versioned; inference precision is unchanged. Public candidate generation/cache and inference calibration still use their reference paths until migration/switching.
- Added cancellation tokens independent of PyO3 locks and fail-fast one-shot rank supervision with bounded log draining.
- Fixed the existing single-GPU finalizer's missing global keep count, which prevented natural-growth calibration from observing a full frontier. Added partial/full single-GPU counter checks and retained multi-rank final-union checks.
- Validation evidence is recorded separately under test_results/refactor_20261107; full-plan and architecture-wide acceptance remain pending.

## 2026-11-07 - Rust inference calibration control
- Added rank-local inference pools and PyO3 ownership. Ensemble probes retain weights/processes across compatible batches; existing native and LibTorch MLP probes preserve their one-shot protocol. All ranks start before readiness, and only a failed independent scorer rank restarts.
- Moved coarse/refinement microbatch search, capacity bootstrap, fresh shared-proposal checks, all-rank timing statistics, telemetry and reserve calculation into Rust. Preserved numerical kernels and the existing inference-only selection threshold; CUDA promotion still requires paired full-step evidence.
- Reproduced and fixed retained-session deadline poisoning and incorrect OOM classification of failed quality receipts. Added active Python-thread cancellation, phase cleanup and telemetry serialization compatibility.
- Native MLP, LibTorch MLP and ensemble control comparisons passed on one/two RTX3060, including partial final batches and independent numerical oracles. Evidence and final source-bound checks are recorded under test_results/refactor_20261107. Public configuration/cache migration, CNOT orchestration, CUDA optimization and final switching remain in progress.

## 2026-11-07 - Rust search session and typed result ownership
- Added typed permutation validation/replay and bounded incremental terminal/collection parsing. Rust now owns compatible session resources, model/geometry restart, all-rank outcome validation and atomic results through PyO3.
- Fixed a reproduced preparation/search timeout reset and valid collection diagnostic paths containing spaces. Cancellation uses an independent token; normal group replacement does not cancel the owning session.
- Remote Rust41 CPU tests, PyO3 cancellation, 24 native/LibTorch eager one/two-GPU cases and independent baseline eager comparisons passed. All-rank cleanup passed for cancellation and one missing-library rank; exact widths, legal collections and retained/replaced PIDs were checked.
- Corrected earlier STAGE2/STAGE3_HOST wording: their LibTorch-linked session libraries used native_cuda_graph, because selecting libtorch_eager requires an explicit environment setting. Historical raw evidence is unchanged. The new tests assert the actual executor in every rank log.
- Public orchestration/default migration, CNOT, CUDA optimizations and full hardware/performance gates remain in progress; see test_results/refactor_20261107/STAGE3_SESSION.md.

## 2026-11-07 - Typed Rust configuration and complete inference jobs
- Added strict NativeOptions, streaming SHA-256 artifact checks, complete inference-job preparation and validated atomic profile caches. Cache hits recompute statistics from complete rank receipts and telemetry; changed weights/helpers and released capacity pressure cannot reuse old acceptance.
- Added the thin rust_control adapter. Calibration samples use a versioned deterministic walk from the graph center, with sample/artifact/graph/device identity in the new key. Existing specialized kernels and precision are unchanged.
- Reproduced and fixed normal probe teardown setting its parent's cancellation token. Reproduced delayed notebook SIGINT; coarse PyO3 search/inference jobs now check Python signals while Rust works, cancel owned processes and join cleanup before KeyboardInterrupt.
- Remote verification: Rust44 CPU tests, Python527pass9skip, actual native/LibTorch/ensemble jobs and cache failure/reuse on1/2RTX3060; actual public-model/options adapter; 24search-session cases and all-rank failure cleanup rerun. Raw evidence is in test_results/refactor_20261107/STAGE3_CONFIG.md.
- Public default switching, downstream policy/cache transfer, CNOT control, CUDA optimization and other GPU-family/performance acceptance remain in progress.

## 2026-11-07 - Rust downstream jobs and exact-capacity profile caches
- Added typed component contracts, native-admitted bounded candidates, all-rank telemetry, original service-work arithmetic and one recorded baseline retry. Rust now owns complete downstream calibration jobs and validated caches through a thin PyO3/Python adapter.
- Optional full-step mode measures the exact requested width, one warmup and five repeats; only its bounded measured shortlist selects a full-step winner. Numerical kernels are unchanged. The new full-step selector uses 5% separation; CUDA promotion still needs separate paired evidence.
- Reproduced/fixed scalar input-row versus native-parent conversion, cancellation during planner startup, normal pipeline cleanup poisoning its parent, hidden native capacity diagnostics and one-ULP JSON timing-cache drift. Enabled the pinned serde_json float_roundtrip feature.
- Native and actual LibTorch eager one/two-RTX3060 comparisons cover primary/repeated cached jobs, corrupted-cache rejection, the real public adapter and all-rank cancellation. CPU and raw GPU evidence are recorded in test_results/refactor_20261107/STAGE3_DOWNSTREAM.md.
- An initial 4096-parent/5120-capacity full-step failure also reproduces on frozen 0eafb193. It remains documented as an admitted-but-not-operational profile; comparison fixtures explicitly reserve 131072 metadata entries without shrinking the beam. Maximum/working-width control, CNOT, CUDA optimization and public default switching remain pending.

## 2026-11-07 - Rust frontier admission and working-width control
- Connected inference calibration, exact-width admission and downstream calibration in one cancelable Rust job. Numeric widths are preserved; symbolic max separates memory admission, exact-width inference refinement and bounded fresh complete-step working-width comparisons.
- Added finite multiset/VRAM ceilings without claiming an orbit proof. The existing calibration fixture-state budget remains separate from the optional explicit diagnostic descriptor ceiling; it never silently caps search width.
- Reproduced and fixed acceptance of an incomplete admission rank cohort. The regression covers rank count and ordering; per-rank runtime errors remain errors, not implicit capacity reductions.
- Rust CPU, Python regression and actual native CUDA Graph/LibTorch eager one/two-RTX3060 tests are recorded in test_results/refactor_20261107/STAGE3_FRONTIER.md. Public launch switching, CNOT orchestration, CUDA optimization and other hardware/performance gates remain pending.

## 2026-11-07 - Experimental full-hash CUB radix sorting
- Added an opt-in full-128-bit radix path for Stream 3/4 and final logical union, preserving deterministic reduction and using existing disjoint A/B arrays with actual CUB scratch queries. The default comparison-sort path remains selected; the Python build explicitly requests the default.
- Both builds passed Stream 3/4, graph, union, static-memory and stitched GPU checks on RTX3060. Memcheck reported zero errors/leaks for radix, Stream 3 and captured union. A standalone test compiled for SM75/80/90/120; those are not full-engine or physical-hardware acceptance.
- Five alternating full-step pairs at width4096 with the same FP16 fixture and two RTX3060 showed a median 13.99% throughput reduction and 365,568 additional planned bytes per GPU for the radix variant. It is not promoted. Raw evidence and measurement limits are in test_results/refactor_20261107/STAGE4_RADIX128.md.
- An independently repeated five-pair series after reconnecting to Vast showed a median 14.73% throughput reduction on the same fixture. Repeat logs, hashes and telemetry are in test_results/refactor_20261107/stage4-radix128-repeat-3060/; the experimental default remains off.

## 2026-11-07 - Experimental Stream 4 index sorting
- Added an opt-in radix path sorting Hash128 plus uint32 indices while keeping compacted CandidateMeta stationary and gathering winners to the survivor arena. The index arrays share the existing value-B allocation; actual CUB temporary storage is queried.
- Compute Sanitizer first exposed invalid metadata reads from inactive CUB lanes. The reducer now checks the live device count, which stays distinct from the CUB unique-count output; memcheck and initcheck passed the scoped GPU tests after the fix. The earlier benchmark of the faulty build is retained but invalid for promotion.
- The combined Stream 3 radix/Stream 4 index build measured median throughput -11.66% versus default and +3.43% versus full-metadata radix on five pairs. Independent Stream 4 index sorting with default Stream 3 measured +2.15% on five pairs, then +2.24% with 8 MiB lower sampled peak per GPU on five more pairs. The 25 ms sampling interval can miss short peaks. A profile-specific selector is still needed before default promotion; the path remains off. See test_results/refactor_20261107/STAGE4_INDEX_SORT.md.

## 2026-11-07 - Experimental Stream 4 scans
- Added an opt-in CUB DeviceScan/BlockScan path with stream-ordered shared scratch, device-resident totals and actual temporary-size queries. Public builds explicitly leave it off.
- Scan-only and scan plus index sorting passed scoped two-RTX3060 tests including large CPU-oracle union, graph replay, insufficient scratch, memcheck, initcheck and primitive racecheck. Repeated native sessions matched the reference on one and two GPUs.
- Five paired complete-step trials measured median +0.28% for scan alone and +0.94% with index sorting, with no change in planned memory. Both fall short of the speed-promotion threshold. See test_results/refactor_20261107/STAGE4_SCAN.md for exact scope and evidence.

## 2026-11-07 - Experimental stable owner partition
- Added opt-in stable owner radix partitioning with existing same-stream scratch, sentinel-aware key width and the original single-rank production path. Scratch is queried and validated before enqueue; the first underallocated CUB attempt was corrected after sanitizer exposed CUDA API errors.
- CPU-oracle tests cover 1 through 256 logical owners and graph replays on each physical RTX3060; one/two-GPU repeated sessions match the reference. Scoped memcheck and initcheck passed, including combined owner/index/scan memcheck. This is not 256-GPU, CNOT or multi-node acceptance.
- Five paired full-step trials show median -0.20% throughput and unchanged planned memory on this profile, so the option remains off. In this initial build a corrupted device count exceeding capacity trapped without exact required/available counts. Details: test_results/refactor_20261107/STAGE4_OWNER_RADIX.md.

## 2026-11-07 - Owner count overflow diagnostics
- The experimental owner router now records count-capacity failure in the existing fatal trace, preserving an earlier failure and returning exact required/available counts before NCCL exchange. It suppresses output on failure without poisoning the CUDA context.
- On each RTX3060, capacity+1 and UINT32_MAX count injection passed in eager and CUDA Graph tests for one/two logical owners, followed by 1,848 normal checks. Scoped memcheck/initcheck, combined owner/index/scan memcheck, the default regression and repeated one/two-GPU native sessions passed. End-to-end injected multi-rank failure and public error text remain untested.
- Five fresh full-step pairs measured median -0.13% throughput with unchanged planned memory. The option remains off; details and limitations: test_results/refactor_20261107/STAGE4_OWNER_COUNT.md.

## 2026-11-07 - Experimental bounded score-histogram radix bits
- Added a sentinel-aware score sort width shared by Stream3/4 launches and actual CUB scratch queries. The current 19-bit path is opt-in; Hash128 and inference precision are unchanged.
- CPU-oracle histogram, eager/graph, one/two-GPU session, default and combined-option regressions passed. Scoped memcheck passed. Uninitialized-scratch initcheck reported the same 9,912 CUB scratch reads with score bits OFF and ON; explicit test scratch initialization removed the reports. Production arena initialization and semantic impact remain to validate.
- Five paired full-step trials measured median +0.85% throughput and unchanged planned memory, below the speed threshold. The option remains off. See test_results/refactor_20261107/STAGE4_SCORE_BITS.md.

## 2026-11-07 - Initialize CUB arena scratch once
- The static arena now merges and initializes only its CUB scratch extents before handing them to nonblocking execution streams; initialization errors release the allocation. Low-level callers supplying their own scratch retain initialization and lifetime responsibility.
- The old allocator reproduced uninitialized CUB scratch reports through production arena pointers. Default and all-five-experimental-option builds passed scoped arena, Stream4, union and graph initchecks on both RTX3060, score memcheck, and repeated native sessions on one/two physical GPUs. This is not whole-application or other-architecture initcheck acceptance.
- Five matched complete-step pairs each on one/two GPUs measured median throughput changes +0.29%/-0.06% and unchanged planned memory. Process-wall timings are not isolated initialization timings. See test_results/refactor_20261107/STAGE4_CUB_INITIALIZATION.md.

## 2026-11-08 - Roll back failed GPU arena construction
- A rollback guard now owns a new static allocation until all layout views and CUB initialization succeed. A rejected reallocation preserves the existing arena.
- The old union-layout error reproduced the retained allocation. Two invalid layouts and reallocation preservation pass; memcheck/initcheck report zero errors and zero leaked bytes on each RTX3060. Repeated Rust/native sessions matched the prior runner on one/two GPUs. Real OOM and injected multi-rank allocation failure remain untested. See test_results/refactor_20261107/STAGE4_ARENA_ROLLBACK.md.

## 2026-11-14 - Experimental session-owned NCCL identity verification
- Added an opt-in per-session cache of the conventional full NCCL SHA-256 and fixed-size block digests. Repeated checks reread and hash all library blocks in parallel; detected concurrent file changes fail. The header is checked independently and the default session path is unchanged.
- CPU tests cover empty and partial blocks, same-size edits with restored mtime, replacement, size changes, detectable concurrent mutation, prepared-runner mismatch and session reuse. The 16 MiB Windows CPU diagnostic is not a GPU-search performance result. GPU validation and performance promotion remain pending. See test_results/refactor_20261114/STAGE_NCCL_IDENTITY.md.
## 2026-11-14 experimental MLP loaded-buffer binding

- In the opt-in explicit-MLP session with CUTLASS, native ranks now bind the
  parsed manifest and already-owned host tensor vectors to the Python artifact
  SHA-256 before GPU upload. Tensor names follow Python's lexicographic order;
  64 KiB input chunks keep SHA scratch bounded without a second full tensor copy.
- A new group requires one matching startup receipt per rank. Missing, wrong or
  duplicate receipts fail the call, including with an older external runner.
  Compatible repeated calls reuse the original rank verification, not a new
  GPU-memory hash. The ordinary path, LibTorch, ensembles and CNOT are not
  covered by this binding.
- CPU RED/GREEN checks under `test_results/refactor_20261114/mlp-binding-3060/`
  include Python/C++ digest equivalence, FP16/BF16, LayerNorm, lexical block
  ordering, buffers above the scratch size, source-file and owned-buffer
  mutations, and twenty mocked launch/receipt and session-transition cases.
- Linux CTest and the integration suite passed (627 passed, 9 skipped). CUDA
  built for SM86; 20 bounded one/two-RTX-3060 cases and four mode-transition
  groups passed with Python and Rust session owners. Five paired repeat calls
  on one synthetic fixture gave a median -2.91% wall-time change, below the
  5% promotion threshold. This is not a saturated GPU-step benchmark; other
  hardware, model families and calibration overhead remain unverified. See
  `test_results/refactor_20261114/STAGE_MLP_LOADED_BUFFER_BINDING.md`.

## 2026-11-17 - GitHub-hosted Vast status and descriptor report

- Added Actions workflows for read-only status of fixed Vast instance 61080396 and for reading a sanitized descriptor report only when that existing instance is running. The report contains status, file metadata and selected error-marker flags; raw remote logs remain on Vast. No instance start/stop or GPU work was performed.
- [Status run 44126220804](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44126220804) and [descriptor-report run 44129242777](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44129242777) completed successfully. The remote descriptor log predates the current uncommitted descriptor-validation change and still contains the earlier async assertion markers; it does not validate that change.
## 2026-11-17 - Pinned source transfer to Vast

- The GitHub-hosted workflow checked out published source commit
  f83b5b658358c5df5067f03dfbb970b2e7551e4e with a code-only sparse checkout,
  archived CMake/Rust/CUDA/native source and integration code, and transferred
  it through the Actions runner to the already-running Vast instance. The
  remote SHA-256 and byte count matched; required source files were present.
  The archive was unpacked into an isolated, run-specific directory and no
  build or beam search was launched.
- Successful transfer run 44132499312:
  https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44132499312
  uploaded a 420-byte receipt artifact with SHA-256
  c4795b7ae89362009ad95608752edb86e3d124327c8e3ddb0fc7428dd406109c. The
  transferred archive was 811626 bytes with SHA-256
  3dc6a3ecc3c7e7dcd80fda306d73ee010cd0efd959e2bdededb2daad6100126a;
  the report counted 457 files.
- The first SCP attempt failed before remote verification (exit 255). The
  workflow now waits for the runner-generated key to propagate and streams the
  archive over SSH stdin. This establishes the GitHub-to-Vast source path only;
  it does not validate a build or the local uncommitted descriptor fix.

## 2026-11-18 - Rust-owned CNOT control collectives

- Added a versioned additive C ABI for int64 NCCL all-reduce, fixed all-gather,
  and broadcast on the existing Stream 5 communicator. Rust/PyO3 checks dense
  CUDA tensor type, dtype, device, element count, current stream, and borrowed
  buffer lifetime; CNOT rank-wide selection uses the Rust path only when every
  rank advertises both exchange and collective ABIs. Older binaries retain the
  Torch fallback.
- Routed CNOT counters, fixed metadata gathers, and reconstruction history
  broadcasts through that selected transport. NCCL-ID bootstrap and
  failure-only diagnostics remain on the caller's process group.
- Added a two-rank NCCL probe and GitHub Actions validation. The initial red
  test run (44190504858) reported the expected 3 failures and 21 passes before
  the implementation. CPU run 44199043726 passed Rust formatting, 3 core CNOT
  tests, 4 CNOT runtime tests, the PyO3 cargo check, 24 Python tests, and probe
  syntax compilation. Full output is retained in its Actions artifact.
- Vast status run 44199043691 confirmed instance 61080396 was exited/stopped;
  it skipped the CUDA build and two-rank NCCL test without changing instance
  state. Physical multi-GPU acceptance remains pending. See
  `test_results/refactor_20261118/cnot-control-collectives/GITHUB_STATUS.md`.

## 2026-11-18 - Rust-owned CNOT scorer control

- Added a thread-bound scorer owner in `beam-native-sys` and a PyO3
  `TQPScorer` adapter. The Rust path owns the dynamic library/context lifetime
  and the existing scalar-ABI chunk loop. It keeps the same
  `min(inference_batch, max_packet_rows)` subpacket partition and call order;
  the experimental `mgbs_tqp_score_many` path remains opt-in.
- PyTorch remains the owner of CUDA buffers. The native boundary checks tensor
  type, dtype, device, shape, contiguity, byte extents, and buffer overlap
  before borrowing device addresses. Python API and a reference fallback are
  retained for older or absent extensions.
- GitHub CPU run 44210146770 passed Rust formatting, 3 `beam-core` CNOT tests,
  4 `beam-runtime` CNOT tests, 4 native-sys scorer tests, `cargo check` and
  `cargo build` for `beam-python`, 28 Python CNOT compatibility tests, and
  NCCL probe syntax compilation. The Python suite built/imported the actual
  PyO3 extension and checked public dispatch and bool argument behavior. Full
  output is in artifact 14385263763 (SHA-256
  `ebd343ddcdf9f7ca0a62ecd14b994293b3b720d2fb7c41f47565a57a3258b212`); see
  `test_results/refactor_20261118/cnot-tqp-scorer/GITHUB_STATUS.md`.
- This is CPU and compile evidence only. The checkpoint scorer library was not
  loaded; CUDA scoring, NCCL execution, and physical multi-GPU acceptance
  remain unverified. This change does not alter search decisions or model
  numerics.

## 2026-11-18 - Rust CNOT scorer dispatch

- CNOT's Rust-owned stage and batch paths now call the built-in PyO3 TQP
  scorer's Rust methods directly. Other scorer implementations retain the
  `parent_scores` / `child_values` Python protocol. The existing score
  validation, call order, and chunking are unchanged.
- Added the dispatch module to hosted Rust formatting and CNOT CPU validation.
  The first action found only rustfmt diffs; the correction is in commit
  `6bbd8ada`. Hosted run 44213429756 then passed Rust formatting, 3 core CNOT
  tests, 4 runtime CNOT tests, 4 native-sys scorer tests, cargo check/build,
  and 33 Python tests, including two `run_cnot_native` CPU stage tests for
  Python scorer fallback. Artifact 14387022871 has SHA-256
  `c2c1b383d37a36357db0ed4f057661a24fb04644646538dd061249945a6e26d7`.
- The built-in TQP direct call was compile-checked but not invoked. The
  current Vast status workflow confirmed the instance stopped and skipped the
  GPU build and two-rank NCCL test without changing its state. CUDA scorer and
  physical multi-GPU CNOT acceptance remain unverified.

## 2026-11-18 - Rust-owned permutation depth budget

- Moved `native_depth_budget` arithmetic into `beam-core` and exposed it via
  the PyO3 extension. Python still validates inputs and keeps its fallback for
  older/missing extensions and integers outside Rust's `u64`/`u32` parameter
  ranges. Search depth and touch-BFS semantics are unchanged.
- The first hosted run exposed the oversized-integer boundary: 36 Python tests
  passed and the two new boundary cases failed against the unguarded FFI call.
  The range guard was added and hosted run 44216092965 then passed formatting,
  3 core CNOT tests, 4 core search tests, 4 runtime CNOT tests, 4 native-sys
  scorer tests, PyO3 extension build/check, 38 Python tests, and NCCL probe
  syntax. Artifact 14388107367 has SHA-256
  `0ed3aa03c26239652506b4a6cab4896b74a15eaf031123c7b1401b1c172cbfee`.
- This is CPU/control-flow evidence only. No CUDA kernel, NCCL collective,
  physical multi-GPU search, or performance path was run.

## 2026-11-18 - Rust-owned touch-BFS entry budget

- The duplicate-free geometric host-entry bound now runs in `beam-core` and
  is shared by Rust `search_budget` and the optional PyO3 adapter. Python keeps
  input checks, maps a rejected cap to the existing `NativeUnavailable`
  message, and falls back for old extensions or values outside the native
  parameter ranges. The 32-generator suffix rule remains in the search
  contract; this changes ownership only.
- Hosted run 44218335727 passed Rust formatting, 3 core CNOT tests, 5 core
  search tests, 4 runtime CNOT tests, 4 native-sys scorer tests, PyO3
  build/check, 44 Python tests, and NCCL probe syntax. Artifact 14388638855
  has SHA-256
  `c2e9cc7ba1cf5d4922bae8eb9c127679800f2ba2846d353ab30d643db557c8fc`.
- First, the new API tests failed as intended on run 44217173408 (39 passed,
  2 failed); run 44218056948 caught only rustfmt layout and the correction is
  in `922b1f7e`. The final run above is green. No CUDA kernel, NCCL collective,
  or physical multi-GPU search ran.

## 2026-11-18 - Rust validation for NativeOptions

- Public `NativeOptions` construction now calls the `beam-core` validator through optional PyO3. Python still normalizes paths and devices and keeps its validator for old, missing, or unloadable extensions and values outside Rust's fixed-width ranges. Existing public error messages remain stable.
- The option descriptor omits `experimental_index_sort` only when the value is exactly `False`; invalid falsey values such as `None` are passed to Rust and rejected. The CPU workflow now covers the complete backend/control test files and triggers on `rust_control.py` changes.
- Run 44223487738 exposed `None` being dropped (1 failure, 119 passed). Test-only run 44224826858 confirmed the missing extension-import fallback (1 failure, 120 passed). Final hosted run 44225114516 passed Rust CNOT/search/runtime/native-sys checks, PyO3 check/build, 121 Python tests, and the two-rank NCCL probe syntax check. Artifact 14391754257 SHA-256: `44e646ba4a7c5d264baad8c54bd86fce06988ea5b2a97ea71f3fe308160ef0aa`.
- This is CPU/control-flow evidence. CUDA kernels, NCCL runtime collectives, physical multi-GPU search, and performance were not run.

## 2026-11-18 - CNOT optional extension import fallback

- The shared Rust extension loader now tries both supported module names and maps missing modules or import/dependency failures to `NativeUnavailable`. CNOT `validation="auto"` and `engine="auto"` can use the Python compatibility path when the optional extension cannot load; explicit `rust` modes still fail closed.
- Hosted run 44227238318 was the intended red check (1 failed, 121 passed). Run 44227654569 passed Rust CNOT/search/runtime/native-sys checks, PyO3 check/build, 122 Python tests, and the two-rank NCCL probe syntax check. Artifact 14393946014 SHA-256: `6481aa6e633831229d7169fa1464102c17f07107d4148cb4e77f7a62c7608726`.
- This verifies CPU/control-flow compatibility only. No CUDA kernel, NCCL collective, physical multi-GPU search, or performance run was performed.

## 2026-11-18 - Rust dispatch for CNOT control collectives

- `World.sum`, `World.gather_fixed`, and `World.broadcast` now dispatch through
  PyO3 functions in Rust when the extension exposes the APIs. Rust preserves
  the existing selected `world.transport` or `torch.distributed` operation;
  tensors and the transport implementation remain Python/PyTorch-owned. The
  single-rank `gather_fixed` path and fallback for older extensions are intact.
- Test-first run 44356451066 failed as intended: the new ABI/dispatch checks
  failed (2 failed, 138 passed, 2 skipped). Run 44357542114 then passed Rust
  CNOT checks, `cargo check`/`cargo build -p beam-python`, 140 Python tests,
  and NCCL probe syntax. Artifact 14452657387 SHA-256:
  `6390bd7da9d7cf023985740c579ca03497d1c865e74d52b4bbae29c6de636d1d`.
- Vast status run 44357542194 confirmed the existing instance was stopped and
  skipped GPU transfer/build/NCCL without changing its state. This is
  CPU/control-flow evidence only; physical CUDA/NCCL and multi-GPU CNOT remain
  unverified. Details: `test_results/refactor_20261118/cnot-control-collectives/GITHUB_STATUS.md`.

## 2026-11-18 - Rust dispatch for CNOT row exchange

- `World.exchange` now calls the versioned PyO3 exchange API when available;
  older extensions retain the existing Python path. The Rust helper keeps the
  chosen transport and collective order, preserves the `native_packet` flag,
  and leaves the existing direct ABI and native-stage packet path intact.
- Test-first run 44359803108 failed as intended: the new API/dispatch checks
  failed (2 failed, 143 passed, 2 skipped). Run 44360850283 passed Rust CNOT
  checks, formatting for the exchange module, PyO3 check/build, 145 Python
  tests, 2 skips, and NCCL probe syntax. Its Gloo exchange test covers
  rank-order, empty splits, and reuse. Artifact 14456055164 SHA-256:
  `308e8d044b951c8f699acee98b54f07a97fea756da29d8cb100c66ef7c6ab35e`.
- Vast status run 44360849887 confirmed instance 61080396 was exited and
  skipped the CUDA/NCCL run without changing instance state. No physical
  multi-GPU runtime was verified. Details:
  `test_results/refactor_20261118/cnot-control-collectives/GITHUB_STATUS.md`.

## 2026-11-18 - Rust selection of CNOT transport

- `World.start_transport` now delegates transport selection to a PyO3 Rust
  function when available. Rust preserves the all-rank support vote for
  `prefer_rust=True`, chooses Rust only when every rank supports it, keeps the
  Python `Stream5` fallback, and skips the vote when Rust is not preferred.
  Older extensions retain the previous Python implementation. No search,
  collective-order, communicator, or wire-protocol behavior was changed.
- Test-first run 44362409466 failed on the new ABI/selection assertions as
  expected (3 failed, 146 passed, 2 skipped). Run 44363394178 passed Rust
  formatting/checks, PyO3 build, 149 Python tests (2 skipped), and NCCL probe
  syntax. Artifact 14455567451 SHA-256:
  `0f42f0b4e21e7e292538f4681f56b6d3e0bcbb1e8a5728fcd3f9108c56c88724`.
- Vast status run 44363394309 confirmed the existing instance was stopped and
  skipped transfer/build/NCCL without changing its state. Physical CUDA/NCCL
  and multi-GPU CNOT runtime remain unverified. Details:
  `test_results/refactor_20261118/cnot-control-collectives/GITHUB_STATUS.md`.

## 2026-11-18 - Rust all-rank vote for native CNOT stages

- `World.all_ranks_support` now delegates the optional native-stage availability
  vote to PyO3 Rust when the ABI exists. Rust keeps the same `World.longs` and
  `World.sum` reduction and accepts the stage runner only when every rank
  supports it. Old extensions retain the former Python vote. The Rust CNOT
  controller and stage execution paths are unchanged.
- Test-first run 44364623412 failed on the four expected new ABI/dispatch/vote
  assertions (4 failed, 149 passed, 2 skipped). Run 44365202646 caught only a
  rustfmt layout issue before compilation. Run 44365545571 passed Rust
  formatting/checks, PyO3 build, 153 Python tests (2 skipped), and NCCL probe
  syntax. Artifact 14456324597 SHA-256:
  `95979ec2f47c2b6d6b5813d88aabaa38e24f654d71247bc1e1613407e06c0293`.
- Vast status run 44365545577 confirmed the existing instance was stopped and
  skipped transfer/build/NCCL without changing its state. Physical CUDA/NCCL
  and multi-GPU CNOT runtime remain unverified. Details:
  `test_results/refactor_20261118/cnot-control-collectives/GITHUB_STATUS.md`.

## 2026-11-18 - Rust CNOT rank-input consensus

- `World.require_consensus` now calls a PyO3 Rust helper to perform the
  existing `World.longs` plus `World.gather_fixed` comparison and preserve the
  same mismatch error. Python still computes the fingerprint; older extensions
  keep the equivalent Python comparison. No collective order, rank layout or
  search decision changed.
- Test-first run 44367678468 failed on the four new ABI/delegation/fallback
  cases as expected (4 failed, 153 passed, 2 skipped). Run 44368166271 stopped
  on one rustfmt layout difference before compilation. Green run 44368563645
  passed Rust formatting/checks, PyO3 build, 157 Python tests (2 skipped), and
  NCCL probe syntax. Artifact 14457527088 SHA-256:
  `1a7a6d4a6394a24368e155307cc5cdc31f9140a6947b987a8aef1cc231c3abdc`.
- Vast status run 44368166135 confirmed the existing instance was stopped and
  skipped transfer/build/NCCL without changing its state. Physical CUDA/NCCL
  and multi-GPU CNOT runtime remain unverified. Details:
  `test_results/refactor_20261118/cnot-control-collectives/GITHUB_STATUS.md`.

## 2026-11-18 - Rust CNOT top-level orchestration

- The Rust controller now owns per-call admission coordination, input
  consensus, transport selection, and native-versus-compatibility stage
  dispatch. Python remains the public adapter, model/scorer callback boundary,
  and old-extension/reference fallback. Collective order and stage arguments
  are pinned by controller tests; no search algorithm or precision policy was
  changed.
- Test-first run 44390118399 failed on the two expected missing-controller
  assertions (157 passed, 2 skipped). Two subsequent runs exposed and fixed
  missing members in the test fixtures; Rust checks and PyO3 build passed on
  both. Green branch run 44392740368 passed Rust formatting/checks, PyO3
  build, 159 Python tests (2 skipped), and NCCL probe syntax. Main run
  44393853310 passed the same checks. Artifacts 14469202830 and 14469468491
  SHA-256: `b2fb5768a81122c46f8629dda235e4f323eb31937ab7469b665ef7daaf192323`
  and `d6b88e8a2fb27c25555f028d35235fbac0a2202f7f3eb00601c14939cc55076b`.
- This is CPU/control-flow evidence. Physical CUDA/NCCL and multi-GPU CNOT
  runtime remain unverified. Details:
  `test_results/refactor_20261118/cnot-search-runtime/GITHUB_STATUS.md`.

## 2026-11-19 - Rust CNOT input preparation

- Added a PyO3 adapter that validates the packed CNOT tensor geometry and
  search controls, creates the same detached contiguous snapshot, serializes
  its little-endian bytes, and constructs the beam-core owned `CnotMatrix`.
  The Python API delegates to it when the extension supports the method and
  retains the old-extension and Python-reference paths. Search behavior,
  scorer callbacks, and collectives are unchanged.
- Branch run 44397470650 passed Rust checks and formatting, the PyO3 build,
  CNOT CPU compatibility tests, and NCCL probe syntax. Hosted artifact
  14472066732 has SHA-256
  `6dd94319971341e455a170f266fe490ee89318f04c9363a53d31529777b6cac7`.
  Published main run 44398322200 passed the same checks at
  `ddc855d734c0897de6af6efef48a6d711a0be998`; artifact 14470868398 has
  SHA-256 `6cc1a5ca4352fd567e190a47ebbbbdb45f67f467fb0c56746c3a69ef1d72424b`.
  The CPU workflow does not verify physical CUDA/NCCL or multi-GPU execution.
  See `test_results/refactor_20261118/cnot-input-prepare/GITHUB_STATUS.md`.

## 2026-11-19 - Rust CNOT goal in the compatibility adapter

- The CNOT compatibility stage now calls the existing Rust goal helper when
  the PyO3 extension exposes it. Older extensions still use the Python goal
  path; runtime errors from the Rust helper are propagated and do not retry
  through Python. The local goal rule and rank-wide collective sequence did
  not change.
- Corrected test-first run 44399877219 showed the expected Python-goal call
  (162 passed, 2 skipped). Green branch run 44400257858 passed Rust
  formatting/checks, PyO3 build, 163 Python tests (2 skipped), and NCCL probe
  syntax. Artifact 14471674104 SHA-256:
  `107089bf02bbb70c4184c339f04e57e8a764f496326a2bb3cdc48ba0c7de818e`.
  Main run 44400879269 passed the same checks with 163 passed and 2 skipped.
  Artifact 14474065000 SHA-256:
  `d7393cf17c75f014ce44c81b676103a668e22344df9bb3dde1929a401a21e758`.
  Physical CUDA/NCCL and multi-GPU CNOT execution remain unverified. See
  `test_results/refactor_20261119/cnot-compat-goal/GITHUB_STATUS.md`.

## 2026-11-19 - Rust-owned CNOT admission

- The Rust CNOT controller now resolves the existing engine selector and depth fallback and computes the admission SHA-256 digest while preserving the established Python JSON byte encoding. These steps stay within the existing rank-wide admission vote. Python input preparation and the older-extension compatibility entry remain; search behavior, precision policy, and collective order are unchanged.
- Test-first run 44403506598 failed on the expected missing-controller assertions (161 passed, 2 skipped). Build run 44404591194 exposed two temporary-lifetime compiler errors, fixed by retaining the Python byte owners. Green branch run 44405036760 passed Rust formatting/checks, PyO3 build, 163 CNOT CPU compatibility tests (2 skipped), and NCCL probe syntax. Hosted artifact 14474283716 SHA-256: `137d8273de03b5329ce907b4be891594625bda79c46021831787ebb824771679`.
- Main revision 30498f85222b26b179b6caa8cebc9fda7d6d58d7 passed CNOT CPU workflow 44406246496; artifact 14476060013 SHA-256: `edf29e53984ba3a0d6838183c5d78934b9a882d2a9cbe78594e8b31a3c3fa382`. The read-only Vast workflow 44406246457 confirmed the `VAST_API_KEY` secret works, but instance 61080396 was stopped, so its build and two-rank CUDA/NCCL steps were skipped without changing instance state. Physical CNOT multi-GPU execution remains unverified. Details: `test_results/refactor_20261119/cnot-rust-admission/GITHUB_STATUS.md`.

## 2026-11-19 - Typed Rust owner for CNOT input

- Matching PyO3 extensions now return a frozen `CnotSearchInput` carrying the validated `beam_core::cnot::CnotMatrix`, the existing detached tensor snapshot, and raw input bytes. The Python adapter passes it through without reconstructing `_InputMatrix`; four-item responses from older extensions remain supported. Search, scorer, and collective behavior are unchanged.
- Test-first Actions run 44407881591 failed as expected because `CnotSearchInput` was absent (1 failed, 162 passed, 2 skipped). Green branch run 44408663743 passed Rust checks, PyO3 build, 163 CNOT CPU tests (2 skipped), and NCCL probe syntax. Hosted artifact 14476796556 SHA-256: `2f41fd59fbca781ad91269f8731f06544af62c0a7ba99bea15329eab5424d8ec`.
- Main revision 6db529828b23df978248838f81e7ca183a0388ea passed CNOT CPU workflow 44409374208 (163 passed, 2 skipped); artifact 14475029346 SHA-256: `7a7cb686d19de9292d54324e4f2795136287cbe1b63c4c82413a46d8fdaf850b`. Physical CUDA/NCCL and multi-GPU CNOT execution remain unverified. Details: `test_results/refactor_20261119/cnot-input-owner/GITHUB_STATUS.md`.

## 2026-11-19 - Rust-owned CNOT typed-input admission

- For `validation="auto"` and `validation="rust"`, the Rust CNOT controller now constructs the typed `CnotSearchInput` directly inside the existing rank-wide admission error boundary. Explicit Python validation and old-extension paths still use the Python input-builder callback. Snapshot semantics, search, scorer, collective order and multi-GPU layout are unchanged.
- Test-first Actions run 44420817709 failed only on the new ownership assertion (163 passed, 2 skipped); artifact 14482655492 retains the remote test log. Follow-up run 44421729086 stopped at the Rust formatting gate before compilation; the reported line-wrap was corrected, and that run produced no test result.
- Green branch run 44422140414 passed Rust formatting/checks, PyO3 build, 164 CNOT CPU compatibility tests (2 skipped), and NCCL probe syntax. Artifact 14480614488 SHA-256: `368f3dca736109bbebe750b2a4017cf322f7a545faeee795db16e6c912afba7f`. Physical CUDA/NCCL multi-GPU execution remains unverified. See `test_results/refactor_20261119/cnot-native-input-admission/GITHUB_STATUS.md`.
- Published main revision `eb58643834af29cb9fb39b6d50a37f48e2a53748` passed the same workflow as run 44422680720 (164 passed, 2 skipped). Artifact 14482048754 SHA-256: `c2f1a2384390934e3696748912363bc036b6469f41e75d46242f75c31bebf897`.

## 2026-11-19 - Rust-owned CNOT identity and construction checks

- The Rust CNOT stage owner now reads identity and path replay directly from
  its already-owned packed `CnotMatrix`. Legacy Python input objects retain the
  previous method/tensor fallback. The new core identity check compares every
  packed row and word; search order, CUDA operators, scorer calls, collectives,
  and result semantics are unchanged.
- Test-first Actions run 44425293831 confirmed the intended failure at
  `torch.equal` (164 passed, 2 skipped); artifact 14482909047 contains its
  remote logs. Run 44426096259 caught a singular negative test matrix; the
  fixture was corrected to apply one valid CNOT before rerunning.
- Green hosted run 44426467664 passed Rust formatting/checks, five packed CNOT
  core tests, the Rust CNOT controller tests, PyO3 build, 165 Python tests
  (2 skipped), and NCCL probe syntax. Artifact 14485300094 SHA-256:
  `e11a6d4ccf9ee0203fee83c8a584606c7290ef1e056e5e4f7ed7e1632f9f1c20`.
- Published `main` at `b8615ba19eb6b7fa7e13dd89c7a1d0c4e22fbec8` passed the
  same workflow as run 44427129610 (165 passed, 2 skipped). Artifact
  14484482646 SHA-256:
  `f936f740628e88e7d31006debab4a8b000e3038bd81ad73d053f69d6880af1f7`.
- Existing-Vast status run 44427129602 reported instance 61080396 as `exited`
  and skipped the two-rank GPU test without changing instance state.
- Physical CUDA/NCCL and multi-GPU CNOT runtime remain unverified.

## 2026-11-19 - Типизированный Rust preparation identity

- Rust `SearchSession` хранит подготовленную identity в `PreparationIdentity` с фиксированными полями и сравнивает десериализованные значения. Python больше не передаёт `repr(tuple)`: он сериализует граф, модель, геометрию, устройства, параметры и build metadata в JSON через прежнюю границу PyO3. Совместимые старые расширения принимают JSON-строку, а расширения без optional identity методов сохраняют локальный Python fallback.
- RED: Actions run 44445381559 выявил два сбоя JSON-проверки прежнего `repr(tuple)`; run 44445580149 упал на структурном сравнении переставленных полей. GREEN: run 44446065188 — 79 passed, 1 skipped (artifact 14492743778); run 44446065116 — 174 passed, 2 skipped (artifact 14492993116); run 44446065115 — Rust наборы 5 и 2 passed (artifact 14493132685). Полные логи сохранены в Actions artifacts, сводка — в `test_results/refactor_20261119/rust-preparation-identity/GITHUB_STATUS.md`.
- Это проверка Rust/PyO3 и CPU-контроля. Алгоритм и multi-GPU протокол не менялись; CUDA/NCCL runtime на физических GPU не проверялся. Python всё ещё строит descriptor и запускает подготовительный callback.

## 2026-11-19 - Rust owns reusable preparation result metadata

- Rust SearchSession caches the full preparation callback outcome alongside the typed preparation identity; the first step moved metadata ownership for repeated searches. The follow-up at 9a9064e atomically finalizes the first native outcome in Rust as well, after Python adds persistent-session metadata. Python then skips its duplicate merge/write; old PyO3 extensions retain the Python fallback. Details: test_results/refactor_20261119/rust-preparation-outcome-finalization/GITHUB_STATUS.md.
- RED: runs [44484788342](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44484788342) and [44484788323](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44484788323) exposed the absent result-metadata API, as expected. GREEN: [Rust public owner 44486326212](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44486326212), 79 passed/1 skipped; [depth control 44486326192](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44486326192), Rust test groups 5 and 2 passed; [CNOT CPU 44486326225](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44486326225), 174 passed/2 skipped plus Rust/CNOT and PyO3 checks. Artifacts and digests are recorded in test_results/refactor_20261119/rust-preparation-result-metadata/GITHUB_STATUS.md.
- CPU/PyO3 evidence only. NCCL probe syntax passed, but no NCCL collective or physical multi-GPU run was performed. No Vast instance was started.

## 2026-11-19 - Rust assembles resident preparation identity

- The Rust-owned `NativeSession` path now passes typed identity fields through a PyO3 builder that constructs and serializes `PreparationIdentity` in Rust. Older extensions without the optional builder retain the prior Python JSON builder. Search semantics, rank ordering, CUDA/NCCL stages, and numerical behavior are unchanged.
- Test-only revision `bea7c3c4e8f62c947c5143db2069e8083ceef67e` triggered Rust public-owner and CNOT CPU Actions runs 44509341963 and 44509341981; both concluded failure. Logs were not downloaded, so the failing assertion is not attributed. After implementation, hosted runs [44518629892](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44518629892) (Rust public owner CPU), [44518629802](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44518629802) (CNOT CPU compatibility), and [44518629764](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44518629764) (permutation depth control) completed successfully. Full logs remain in GitHub Actions.

## 2026-11-19 - Rust assembles resident launch identity

- `SessionIdentity` construction and serialization now run in beam-runtime via an optional PyO3 builder; RustResidentWorkers uses it for resident launch. Older extensions retain the previous Python JSON builder. This moves configuration assembly only: search stages, ranks, numerical behavior, and the multi-GPU protocol are unchanged.
- RED: test-only commit `420d39f47d35bc64f214e45e068af0f0097f3984` produced failed hosted runs [python-owner 44524014350](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44524014350) and [cnot-cpu 44524013917](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44524013917). Logs remain remote; exact failing assertions were not inspected. GREEN pending.
- This CPU/PyO3 slice does not verify physical CUDA/NCCL or multi-GPU CNOT runtime.

## 2026-11-19 - Rust inference candidate ordering

- `beam-core` now emits the same coarse and refinement batch order as Python's `random.Random`. Its CPython-compatible MT19937 seeding, `getrandbits`, `randbelow`, and shuffle preserve the existing candidate order. Optional PyO3 methods expose both builders; `autotune.py` calls Rust when available and retains the Python fallback for missing or older extensions. Python still owns calibration sampling, cache handling, statistics, candidate evaluation, ranking and admission. Inference kernels and numerical policy are unchanged.
- RED: Actions run [44540169138](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44540169138) confirmed that the new Rust APIs and public routing were absent. GREEN: [44543777970](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44543777970) passed formatting, scoped Rust tests (26 tests across selected groups), PyO3 build, 194 Python tests (2 skipped), and NCCL probe syntax. Artifact 14540030869 SHA-256: `5a2ee6e102e6eed259bac31c9d04dcc8732442a1734313e54458d386ff1368ff`.
- This is CPU/PyO3 evidence only. No physical CUDA/NCCL multi-GPU run or Vast compute was used. Details: `test_results/refactor_20261119/rust-inference-candidates/GITHUB_STATUS.md`.

## 2026-11-19 - Rust inference probe-time allowance

- The deterministic allowance formula now runs in `beam-core`, exposed through optional PyO3 and used by the public Python helper for ordinary finite timing records. Unsupported inputs, missing/older extensions, and unusual timing types retain the original Python calculation. Probe order, calibration, kernels, and numeric behavior are unchanged.
- RED run [44545990522](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44545990522) showed the missing builder and public delegation. Run [44546747536](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44546747536) caught rustfmt; [44547107689](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44547107689) caught an incorrect test expectation where the 12-second default dominates a 6-second estimate. GREEN: [44547537516](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44547537516), 200 passed/2 skipped, 26 selected Rust tests, PyO3 build, and NCCL probe syntax. Artifact 14541208218 SHA-256: `49c4cc1ca852f3df92eefa24d9195e202d4b3621cc6fc94ddfa68543bea9a0cf`.
- CPU/PyO3 evidence only; no physical CUDA/NCCL multi-GPU execution or Vast compute. Details: `test_results/refactor_20261119/rust-inference-probe-time-allowance/GITHUB_STATUS.md`.

## 2026-11-19 - Rust capacity-cache reuse decision

- The deterministic capacity-cache reuse predicate now runs in `beam-core`, exposed through optional PyO3. Python still gathers current free memory and keeps the same non-capacity short-circuit; ordinary UUID-to-MiB maps go to Rust, while unusual maps or absent/older extensions use the equivalent Python fallback. Key matching and the existing 32 MiB pressure threshold are unchanged.
- RED run [44549277542](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44549277542) executed the new tests and reported 9 expected failures (202 passed, 2 skipped). Run 44548798695 was not accepted as RED evidence because its explicit pytest list omitted the new file; the workflow command was corrected. GREEN: [44550275840](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44550275840), 211 passed/2 skipped, selected Rust tests, PyO3 build, and NCCL probe syntax. Artifact 14543596077 SHA-256: `8a4bea305bd3cdbaa0a00ceb2b3a37c5cfcc4214c1e15a542301d13f44903fd1`.
- CPU/PyO3 evidence only; no physical CUDA/NCCL multi-GPU execution or Vast compute. Details: `test_results/refactor_20261119/rust-inference-cache/GITHUB_STATUS.md`.


## 2026-11-19 - CNOT failure diagnostics use fixed tensor gathers

- Added optional `run_cnot_native_checked_v2`; the Rust controller uses the
  version that gathers diagnostic byte lengths and padded UTF-8 byte tensors in
  rank order after the existing error vote. The public `World.checked` path
  performs one cached capability vote across ranks before selecting v2.
- If any rank lacks v2, the adapter retains the legacy Rust checker or Python
  object-gather fallback. Search stages, scorer order, matrix handling and
  winner semantics are unchanged. The first multi-rank checked call gains one
  capability vote; physical CUDA/NCCL recovery remains unverified.
- RED: hosted run [44560022663](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44560022663) reproduced the Python `all_gather_object` call in the new test (217 passed, 2 skipped; artifact 14546707888). GREEN: hosted run [44562629629](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44562629629) passed Rust formatting/tests/checks, PyO3 build, 219 Python tests (2 skipped) and NCCL-probe syntax. Artifact 14549203505 SHA-256: `53d387fea8c2694bf8b2c5d22aeae5771bf11394a6855ed18dfbf4b6576bc374`.
- The separate existing-instance check [44562629392](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44562629392) found Vast instance 61080396 `exited`; it skipped GPU work and issued no start, stop or rent request. Physical CUDA/NCCL error propagation remains unverified.

## 2026-11-19 - Rust compatibility backend checks typed CNOT root goal

- The Rust `PyStages` backend now checks typed `CnotSearchInput` through the
  existing `search_input_is_identity` helper before invoking the Python
  `root_is_goal` stage method. Python-only input and old stage providers keep
  the previous callback fallback.
- Search selection, matrix and tensor ownership, CUDA operations, scorer calls,
  layer ordering and collectives are unchanged; this transfers one compatibility
  stage decision to Rust without changing search behavior.
- RED: hosted run [44585776893](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44585776893)
  failed on the Python stage callback as expected (1 failed, 218 passed, 2
  skipped; artifact 14559869836). GREEN: hosted run
  [44586229891](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44586229891)
  passed Rust checks, PyO3 build and the selected CPU suite (219 passed, 2
  skipped; artifact 14561312765).
- An earlier test run did not force the compatibility-stage path; the regression
  test was corrected to disable native-stage dispatch before recording RED.
  Physical CUDA/NCCL and multi-GPU acceptance remain unverified. Evidence:
  `test_results/refactor_20261119/cnot-rust-root-goal/GITHUB_STATUS.md`.

## 2026-11-19 - Typed CNOT compatibility goal dispatches in Rust

- For typed Rust `CnotSearchInput`, the Rust compatibility backend now calls
  the shared `cnot_native_goal::goal` helper directly. Python-only input and
  older stage providers retain the existing `goal` callback.
- Goal semantics, tensor operations, error coordination, stage order and
  collectives remain unchanged; the change removes only the Rust-to-Python
  callback on this typed compatibility path.
- RED: hosted run [44589295992](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44589295992)
  failed on the Python goal callback as expected (1 failed, 219 passed, 2
  skipped; artifact 14563611172). GREEN: hosted run
  [44589852872](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44589852872)
  passed Rust checks, PyO3 build and selected CPU suite (220 passed, 2 skipped;
  artifact 14562407176).
- Physical CUDA/NCCL and multi-GPU acceptance remain unverified. Evidence:
  `test_results/refactor_20261119/cnot-rust-goal-stage/GITHUB_STATUS.md`.

## 2026-11-19 - Typed CNOT history edges are assembled in Rust

- For typed Rust `CnotSearchInput`, the compatibility `PyStages` backend now
  copies the owner-held history record into the rank-local tensor, invokes the
  existing `World.broadcast`, and validates the parent/route/owner values in
  Rust. Python-only input and older providers retain the previous callback.
- History record layout, target rank, collective sequence, path reconstruction
  and returned path remain unchanged.
- RED: hosted run [44592673898](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44592673898)
  failed on the Python `history_edge` callback as expected (1 failed, 220
  passed, 2 skipped; artifact 14563163386). An intermediate run
  [44593184432](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44593184432)
  stopped at rustfmt; formatting was corrected. GREEN: hosted run
  [44593458330](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44593458330)
  passed Rust checks, PyO3 build and selected CPU suite (221 passed, 2 skipped;
  artifact 14565280008).
- Physical CUDA/NCCL and multi-GPU acceptance remain unverified. Evidence:
  `test_results/refactor_20261119/cnot-rust-history-edge/GITHUB_STATUS.md`.

## 2026-11-19 - Typed CNOT candidate exchange is dispatched by Rust

- For typed `PyCnotSearchInput`, compatibility `PyStages` now calls the
  existing Rust `cnot_native_exchange::exchange` helper with
  `native_packet=true` and stores the result in `incoming`. Python-only input
  and older stage providers retain the Python callback.
- The helper, candidate rows, owner IDs, rank membership, collective order,
  transport selection, payload representation and search semantics are
  unchanged; this migrates dispatch only.
- RED: hosted run
  [44596999245](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44596999245)
  failed at the Python exchange callback as expected (1 failed, 221 passed,
  2 skipped; artifact 14566162577). An intermediate run
  [44597579671](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44597579671)
  stopped at rustfmt before compilation or tests; formatting was corrected.
  GREEN: hosted run
  [44597954192](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44597954192)
  passed Rust checks, PyO3 build and selected CPU suite (222 passed,
  2 skipped; artifact 14565748082).
- MAIN verification after fast-forward: hosted run
  [44598940460](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44598940460)
  passed the same CPU/PyO3 checks (222 passed, 2 skipped; artifact 14566448047).
- Physical CUDA/NCCL and multi-GPU acceptance remain unverified. Evidence:
  `test_results/refactor_20261119/cnot-rust-exchange/GITHUB_STATUS.md`.

## 2026-11-19 - Typed CNOT construction verification is dispatched by Rust

- For typed `PyCnotSearchInput`, compatibility `PyStages` now calls the
  existing `search_input_verifies_construction` helper. Python-only input and
  older stage providers retain the `CnotStages.verifies_construction` callback.
- Path reconstruction, gate acceptance, rank ownership, search semantics and
  scorer behavior are unchanged; this migrates verification dispatch only.
- RED: hosted run
  [44600364785](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44600364785)
  failed on the Python verification callback as expected (1 failed, 222 passed,
  2 skipped; artifact 14567308560). GREEN: hosted run
  [44600777706](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44600777706)
  passed Rust checks, PyO3 build and CPU compatibility tests (223 passed,
  2 skipped; artifact 14566579277).
- MAIN verification after fast-forward: hosted run
  [44601261874](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44601261874)
  passed the same CPU/PyO3 checks (223 passed, 2 skipped; artifact 14568081625).
- Physical CUDA/NCCL and multi-GPU acceptance remain unverified. Evidence:
  `test_results/refactor_20261119/cnot-rust-verify/GITHUB_STATUS.md`.

## 2026-11-20 - Rust владеет CNOT compatibility redistribution

- Для типизированного `PyCnotSearchInput` Rust adapter перенял сборку winner
  packet, вызов существующего обмена, сортировку received rows, статистику,
  audit и подготовку следующего frontier. Опциональные CUDA hooks и Torch
  fallback используют прежний формат данных; старые stage providers оставляют
  Python callback.
- Алгоритм, transport, collective order, precision и scoring не менялись.
  Новый rank-wide error wrapper не добавлен.
- RED: hosted run
  [44613780854](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44613780854)
  упал только на запрещённом Python callback для qhat/proposal
  (2 failed, 228 passed, 2 skipped; artifact 14574401283,
  SHA256 `b193d6bba0fd599faef7afaedcac9840d40aa4d10564aa0be32270e97c08c6e2`).
- GREEN: hosted branch run
  [44615705900](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44615705900)
  и последующая проверка `main`
  [44616358356](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44616358356)
  прошли Rust checks, PyO3 build и CPU suite (230 passed, 2 skipped).
  Main artifact 14576135151, SHA256
  `e761f9675a1fd90a7c46333635c4a7edf932dfc7a38b4eaa54906e154eaebbc1`.
- Физическая проверка CUDA/NCCL и нескольких GPU остаётся невыполненной.
  Evidence: `test_results/refactor_20261119/cnot-rust-redistribute/GITHUB_STATUS.md`.

## 2026-11-20 - Typed CNOT selection dispatch перенесён в Rust

- Compatibility backend теперь напрямую вызывает существующий
  `cnot_native_selection::select` для typed `PyCnotSearchInput`, записывает
  `ranks/local/new_lex` и не создаёт Python `SelectionStages`.
  Старые providers и Python-only input сохраняют callbacks.
- Сохранены radix schedule, CUDA operators/Torch fallback, score и word keys,
  collectives и ordering; алгоритм и precision не менялись.
- RED: hosted run
  [44618218529](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44618218529)
  подтвердил два вызова запрещённого Python selection callback
  (2 failed, 230 passed, 2 skipped; artifact 14576716078,
  SHA256 `4f540d712ba29f03b084638097b00f3efd6ce858de65e81e4c43cf8c01ebe2a3`).
- Исправлены две промежуточные ошибки: rustfmt в run
  [44618613883](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44618613883)
  и избыточная проверка ещё не созданных output attributes в run
  [44618983198](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44618983198).
- GREEN: branch run
  [44619334509](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44619334509)
  и main run
  [44619796548](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44619796548)
  прошли Rust checks, PyO3 build и CPU suite (232 passed, 2 skipped).
  Main artifact 14575103927, SHA256
  `25aa5f670c65f98d86a40cfa56dcc3926bac67e75e7a1cf8a6ab54f5c2b54bd2`.
- Физическая CUDA/NCCL multi-GPU проверка остаётся невыполненной.
  Evidence: `test_results/refactor_20261120/cnot-rust-selection/GITHUB_STATUS.md`.

## 2026-11-20 - Typed CNOT batch fallback dispatch перенесён в Rust

- Typed compatibility stages больше не вызывают `cnot._expand`,
  `cnot._distinct` или `cnot._scores`. Они используют существующий
  `cnot_native_batch` с прежними Torch/CUDA domain primitives и scorer API.
  У fused CUDA hooks приоритет сохранён.
- Сохранены режимы qhat/proposal, scoring, стабильная сортировка по
  generation/route, state equality, metadata и output tensors. Новые CUDA
  kernels и алгоритмические изменения не добавлялись.
- RED: hosted run
  [44621164737](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44621164737)
  подтвердил четыре forbidden helper вызова (4 failed, 232 passed, 2 skipped;
  artifact 14576679253, SHA256
  `96799f794f77e29c0dfb09e1cb6a3dfc438406f019cd2a266de58963661f98bf`).
- GREEN: branch run
  [44622029810](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44622029810)
  и main run
  [44622384926](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44622384926)
  прошли Rust checks, PyO3 build и CPU suite (236 passed, 2 skipped).
  Main artifact 14575724693, SHA256
  `f80dd0f54adb6e497bfc7b762ec37c7f63df71d4a94054493fea56d521c2c00f`.
- CPU/reference implementations of low-level CNOT primitives still use Torch
  in `cnot_native.py`; this is not physical CUDA/NCCL acceptance.
  Evidence: `test_results/refactor_20261120/cnot-rust-domain-fallback/GITHUB_STATUS.md`.

## 2026-11-20 - Rust владеет CNOT route packing

- Typed CNOT expansion теперь проверяет source rank и тип move tensor в Rust,
  применяет прежнюю проверку uint16 диапазона и формирует тот же int64 route
  lane. Вызов cnot_native.pack_cnot_route удалён; алгоритм, metadata и
  порядок поиска не менялись.
- Первый Actions run 44625061344 не включал новый тест в явный список и не
  считается RED-доказательством. После исправления workflow RED
  [44625601360](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44625601360)
  подтвердил оба ожидаемых callback failure (2 failed, 236 passed, 2 skipped;
  artifact 14578637524, SHA256
  c1b408b8c353228f22e2cc60a12f56ae3345a6acfe3fe5acfefe7881350c9d0e).
- Промежуточный run 44626077505 остановился на rustfmt до сборки и тестов.
  Branch GREEN
  [44627095448](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44627095448)
  прошёл Rust checks, PyO3 build, CNOT CPU suite (238 passed, 2 skipped) и
  NCCL probe syntax. Artifact 14579484694, SHA256
  9a420188f9dd7eed2c20bce4ae48585ecc9f639330e7190a7d7a3eaa7028db65.
- Main GREEN
  [44628217397](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44628217397)
  passed the same checks at commit a82afbb172f1d7dbc68356dd632e9e65e8af3a42
  (238 passed, 2 skipped). Main artifact 14580865888, SHA256
  ea6be9a385dc4d5abb74d2de589adbb4318b39d6bdc8937997b73085465aacba.
- Физическая CUDA/NCCL multi-GPU проверка не выполнялась. Полная запись:
  test_results/refactor_20261120/cnot-rust-route-dispatch/GITHUB_STATUS.md.

## 2026-11-20 - Rust component pipeline-cache receipt validation

- `beam-runtime::pipeline_cache` now validates the public
  `component_calibrated` receipt shape; optional PyO3 dispatch in
  `pipeline_cache.py` keeps Python fallback for missing/older extensions and
  all full-frontier receipts.
- Preserved receipt schema, environment allowlist, plan admission checks,
  complete per-rank measurements, requested beam width, and positive finite
  component/transport timings. Fresh native all-rank memory admission after
  cache validation stays mandatory. No search, profile-selection, rank,
  collective, inference, or GPU behavior changed.
- RED: Actions run
  [44648572656](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44648572656)
  failed because the PyO3 validator was not implemented yet; 248 tests passed
  and 2 skipped. Rust checks and the extension build passed.
- GREEN: Actions run
  [44649100197](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44649100197)
  passed Rust checks, PyO3 build, 249 Python tests (2 skipped), and NCCL probe
  syntax. The follow-up legacy-extension fallback run
  [44649561313](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44649561313)
  passed 250 Python tests (2 skipped), Rust checks, PyO3 build, and NCCL
  probe syntax. Artifact 14590465891, SHA256
  9aff948e3aec5b88173fcb2121cae0789680e1625f037afb1c6185093f24de6b.
- Main run [44650117639](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44650117639)
  at commit `16e7f53c059b6cab7391cfb5aeeb6f167ddd05ce` passed the same
  250-test CPU/PyO3 suite (2 skipped), with artifact 14591345545 and SHA256
  `3a4d15ff588e32941510851fae041bcc304593ad89cd849d5ff57719f4950adb`.
  Companion permutation depth-control run
  [44650117646](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44650117646)
  passed. These CPU/PyO3 checks do not establish physical CUDA/NCCL multi-GPU
  execution. Evidence:
  `test_results/refactor_20261120/rust-pipeline-cache-receipt/GITHUB_STATUS.md`.

## 2026-11-20 - Rust rank-frontier quotas for calibration

The Python calibration helper now uses the existing typed
`beam_core::planning::rank_frontier_counts` through optional PyO3. It preserves
the exact ceiling-difference distribution used by native final exchange,
including uneven widths and zero spare-slot contribution. Older extensions
and unsupported legacy shapes retain the Python formula. No beam width, rank
order, collective, search output, or CUDA operation changed. Hosted RED/GREEN
RED run [44651491013](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44651491013) found the missing PyO3 method; its test import omission was corrected. An intermediate fallback-fixture failure was fixed. GREEN run [44652345373](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44652345373) passed the Rust planning contract, PyO3 build, 260 Python tests (2 skipped), and NCCL probe syntax. Artifact 14590718645, SHA256 `53f863655ff5a1c428a8b1abc4b840065a3708b619c2c42615db7fcec89e72d0`. Evidence:
`test_results/refactor_20261120/rust-rank-frontier-quotas/GITHUB_STATUS.md`.
Main commit `148b2a826a369b7601e904ea409e0cc5301c75c9` also passed the 260-test CPU/PyO3 suite (2 skipped) in Actions run [44652813652](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44652813652); artifact 14592080300, SHA256 `3f519592bd4f5a38140257a025b3fb20d7701064f250b5228c4f4af6308555f7`. Physical multi-GPU runtime remains unverified.

## 2026-11-20 - Rust validation of calibration rank plans

- Added optional `beam-core::planning::validate_rank_plans` dispatch through
  PyO3 and routed ordinary typed calibration plan dictionaries to Rust.
- Kept unknown fields outside the payload. Unsupported Python values, missing
  or older extensions, and Rust rejections retain the existing Python
  validator and error messages, including legacy zero-geometry acceptance.
- No calibration choice, beam width, rank ordering, memory policy, collective,
  search output, or CUDA operation changed.
- RED: [Actions run 44656721403](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44656721403)
  passed the existing CPU/PyO3 suite (261 passed, 2 skipped) and failed the two
  new assertions for the missing Rust dispatch and PyO3 method.
- Branch GREEN: [Actions run 44658594054](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44658594054)
  passed 265 Python tests (2 skipped), Rust checks, PyO3 build, and NCCL probe
  syntax. Artifact 14594831023, SHA-256
  `746312d931ec875bbceeb253b9ca5a316a47cbba788805109fda034eee8ccdd1`.
- Main GREEN: [Actions run 44659065473](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44659065473) at `62feb392b66bed588d66e0f94b4049f82f523c3e` passed 265 Python tests (2 skipped), Rust checks, PyO3 build, and NCCL probe syntax. Artifact 14594144027, SHA-256 `61f754fbc8f51c56ad3d1b4407d630a0350539198b443fc9cfa0b3dc06382e34`.
- Physical CUDA/NCCL multi-GPU execution remains unverified. Evidence:
  `test_results/refactor_20261120/rust-rank-plan-validation/GITHUB_STATUS.md`.

## 2026-11-20 - CNOT goal errors use fixed tensor diagnostics

- Reused the Rust checked-stage UTF-8 diagnostic gather for the CNOT goal error
  path. After the existing goal vote, ranks gather lengths and padded bytes via
  `World.gather_fixed`; successful goal selection and its collective sequence
  remain unchanged.
- The older checked-stage ABI retains its object-gather compatibility path.
  No goal, winner, matrix, scorer, search-depth, or rank semantics changed.
- RED: [Actions run 44660740001](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44660740001)
  passed 265 Python tests (2 skipped); the new goal test failed on the existing
  `all_gather_object` call.
- Branch GREEN: [Actions run 44661053113](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44661053113)
  passed Rust checks, PyO3 build, 266 Python tests (2 skipped), and NCCL probe
  syntax. Artifact 14595053280, SHA-256
  `8057258b53a903863f37a50d27a157a064bcd14ab0f8507daca087d8a3bdddd9`.
- Main GREEN: [Actions run 44661706426](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44661706426) at `080f98fd1968ca8af5fc1a18629efb7ab707b5db` passed Rust checks, PyO3 build, 266 Python tests (2 skipped), and NCCL probe syntax. Artifact 14595702908, SHA-256 `5132784e06da62aa6ee3faeeaff9ca014ddc9ed2865fe439c0fa929b656f1f2c`.
- The test uses a two-rank CPU diagnostic fixture; it does not establish
  physical CUDA/NCCL multi-GPU execution. Evidence:
  `test_results/refactor_20261120/cnot-goal-tensor-errors/GITHUB_STATUS.md`.
- MAIN GREEN: [Actions run 44699705418](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44699705418) at `db9577fcc61a3b57fae1ba0e1e2bc5c579e9b001` passed the same Rust/PyO3 CPU suite (269 passed, 2 skipped, 104 warnings). Artifact 14612416568, SHA-256 `51be4e713b50c58e70e9a7b95ec055ab2f039420715dfe275f4e5d9c8a8ddfa6`.
- The automatically triggered existing-Vast workflow [44699705287](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44699705287) observed a stopped instance and skipped remote build/execution; it did not start the instance. Physical CUDA/NCCL validation remains unverified.

## 2026-11-20 - Public cancellation for Rust resident searches

- Added `NativeSession.cancel()` for an active Rust-engine search. It signals
  the Rust resident owner, handles a request arriving during preparation, and
  closes the session after cancellation. Idle sessions and the Python engine
  return `False`; old Rust extensions without cancellation support keep the
  compatibility behavior.
- Added a two-rank CPU integration test that waits for both resident workers,
  requests cancellation, and verifies the stop marker, both worker exits, and
  empty PID ownership afterward. Search algorithm, scoring, precision, rank
  topology, and GPU code are unchanged.
- RED: Actions run
  [44719860829](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44719860829)
  exposed the missing public method (86 passed, 3 skipped, one expected failure).
- GREEN: Actions run
  [44721590584](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44721590584)
  passed the Rust/CNOT CPU workflow, including 3/3 resident session tests,
  Rust checks, PyO3 build, and 280 Python tests (2 skipped). Artifact 14620999670,
  SHA-256 `3930714e9560a28ec895af213bfafc190a5feed85423371b08f412211cfc8122`.
  Evidence: `test_results/refactor_20261120/rust-session-cancel/GITHUB_STATUS.md`.
- This validates host-side process cancellation only; physical CUDA/NCCL and
  1-/2-GPU runs remain unverified.
- MAIN GREEN: run
  [44722196693](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44722196693)
  at `014a2ffeaab5493dd0e0363c93a034fe09413381` passed again on main:
  Rust/CNOT CPU checks, PyO3 build, and 280 Python tests (2 skipped). Artifact
  14621863125, SHA-256
  `c33942672d0c4a24c3845feb5e733236fb6512b731f6fc8ff79fd3729f706840`.
  The companion public-owner workflow [44722196881](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44722196881)
  also passed. Physical CUDA/NCCL and 1-/2-GPU validation remain unverified.

## 2026-11-20 - Installed Python wheel validation

- Added a GitHub Actions acceptance job that builds the package through its
  PEP 517 backend, installs the produced wheel, checks dependencies, and runs
  the smoke check from outside the source checkout with `PYTHONPATH` unset.
- The check confirmed both `multigpubeamsearch` and the compatibility
  namespace `cayleypy_native`, the public search/model/session exports, Rust
  ABI v1, `TQPScorer`, `InferenceCalibration`, the packaged executable
  `beam-runner`, and its `resident-search-v1` protocol.
- Branch run [44724395099](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44724395099)
  and main run [44727232048](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44727232048)
  passed at commit `e8a819fa6a46845bdffad95857007aa5b374ba04` using hosted
  Ubuntu, Python 3.11, and CPU PyTorch 2.9.0. Main artifact 14623599422,
  SHA-256 `0fd444baec3ae693e891a5fe74fe49b6bbe7d4a7a2ed752f32fcbd740d1739a0`.
- This is installed-wheel and host-side worker-protocol evidence. It does not
  validate CUDA execution, checkpoint scoring, NCCL, or physical 1-/2-GPU
  search. Full logs and the smoke summary are in
  `test_results/refactor_20261120/installed-wheel/GITHUB_STATUS.md` and the
  linked Actions artifact.

## 2026-11-20 - Coordinate CNOT selection completion errors

- Added a two-rank Gloo regression that injects a rank-one exception during
  local result ordering after the global selection collectives. Before the
  fix, rank zero entered the next redistribution collective and received a
  peer-closed Gloo error instead of the original diagnosis.
- The Rust CNOT stage now checks the completed selection result through the
  existing Gloo control group before starting redistribution. Both ranks
  receive the same stage error, and the test confirms that a subsequent Gloo
  all-reduce still works.
- Main run [44796514366](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44796514366)
  passed 302 tests with 2 skipped. Artifact [14659601192](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44796514366/artifacts/14659601192),
  SHA-256 `87ae53a9f6ab336e160eff33ec3d7a1c6b282e5043614ffe03398d5b302e686f`.
- Installed PEP 517 wheel run [44796514465](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44796514465)
  passed installation outside the checkout, `pip check`, public import/export
  checks, and the `resident-search-v1` worker protocol. Artifact
  [14658528465](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44796514465/artifacts/14658528465),
  SHA-256 `8b7d94b169c9f3bf82b95b94a91a7ec94bcbe037fc7ed83b99cee165a55b924e`.
- The read-only existing-Vast job [44796514430](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44796514430)
  found instance 61080396 exited and skipped GPU execution; no instance
  state change was attempted.
- CPU/Gloo and wheel checks do not establish CUDA, NCCL, or physical 1-/2-GPU
  acceptance. Errors inside selection before an internal collective are not
  covered by this completion vote.

## 2026-11-20 - Coordinate CNOT record exchange preparation

- Added two two-rank Gloo regressions for rank-local exchange failures: packet
  construction before the first data collective and receive allocation after
  the count exchange but before payload exchange. Both tests verify identical
  rank-wide errors and a successful follow-up Gloo all-reduce.
- The prior packet-preparation RED run
  [44797521897](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44797521897)
  produced a peer-closed error on the healthy rank. The Rust exchange now
  coordinates packet construction before any rank enters the data-plane
  collective and coordinates fallback payload sizing/allocation before the
  second all-to-all. Existing Stream 5 internal checks remain in place.
- Main CPU run
  [44798848987](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44798848987)
  passed 304 tests with 2 skipped. Artifact
  [14661050464](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44798848987/artifacts/14661050464),
  SHA-256 `eee8773cdfa27fb8e5da41ec48a5afceaf6c8b3a4299520f7328f2c87c504a40`.
- Installed wheel run
  [44798339035](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44798339035)
  built and installed the PEP 517 wheel outside the checkout, passed
  `pip check`, public exports, and `resident-search-v1`. Artifact
  [14660825488](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44798339035/artifacts/14660825488),
  SHA-256 `525eaeeeb26246dc0f4b70675748e9c5fd960818c261e4d46aafb0d157354469`.
- Read-only Vast run
  [44798339107](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44798339107)
  found instance 61080396 exited and skipped physical GPU checks. No
  start/stop/rent request was made.
- These checks do not establish physical CUDA/NCCL behavior. Failures inside
  low-level data-plane collectives remain outside the local preparation votes.

## 2026-11-20 - Coordinate CNOT cancellation across ranks

- Rust checked CNOT stages poll Python signals before their Gloo control-group
  vote. A rank-local `KeyboardInterrupt` is gathered and raised as
  `KeyboardInterrupt` on each rank; ordinary failures remain
  `RuntimeError`. The CNOT goal vote includes a local signal before peers
  continue. The Python compatibility fallback preserves the cancellation type.
- Search order, scorer policy, precision, rank ownership, and NCCL data-plane
  operations are unchanged.
- RED: run
  [44800867787](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44800867787)
  reproduced the issue: rank 1's `KeyboardInterrupt` became a
  `RuntimeError` at `check CNOT root goal`.
- GREEN: main run
  [44802610199](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44802610199)
  at `1f331663cc3368b1e7a357abc5a4d5bb7b390301` passed the Rust checks,
  PyO3 build, CNOT CPU/Gloo suite (305 passed, 2 skipped). Artifact
  14660759672, SHA-256
  `bad79e5f3ee7e6a3a476afef2bb7754992137564399ba4568446710aba16a889`.
- Installed-wheel run
  [44802610026](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44802610026)
  passed PEP 517 build, install outside the checkout, dependency checks and
  public API/protocol smoke checks. Artifact 14662495219, SHA-256
  `ef7d87e8639e80c90c6ec765d599bddb26275c32fd60dedef424a2f974cb81a7`.
- Read-only Vast run
  [44802610013](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44802610013)
  found instance 61080396 exited and skipped the remote CUDA/NCCL step without
  changing instance state. Artifact 14661622569, SHA-256
  `a499417bc4f690641a8e6d43f1c55299b6ffe25819a463e2ea78061885bcf2fc`.
- Physical CUDA/NCCL cancellation, the штатный checkpoint scorer on GPU, and
  physical 1-/2-GPU search remain unverified. Full test and workflow output is
  retained in the linked Actions artifacts.


## 2026-11-20 - Preflight CNOT selection and gathered-row collectives

- CNOT now coordinates rank-local selection preparation, selected-record
  gathering, row-gather buffers and copies, and Stream 5 fixed-gather
  preparation over the existing CPU/Gloo control group before ranks enter the
  next data collective. The Python compatibility path follows the same stage
  order.
- Search order, tie-breaks, scorer calls, numerical policy, candidate IDs,
  ownership, and NCCL payload/collective order are unchanged. This catches
  local preparation failures only while the Gloo control group remains usable;
  process loss or a failed/hung NCCL communicator is not recovered.
- RED: [Actions run 44804192845](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44804192845)
  reproduced a rank-local CNOT histogram preparation exception escaping the
  stage before its peer could proceed safely.
- GREEN: main commit `74f71ccef2895f96b716282e7ee29d9800c564dc`,
  [CNOT CPU/Gloo run 44812797417](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797417),
  passed 308 tests (2 skipped, 104 warnings). Artifact
  [14667597426](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797417/artifacts/14667597426),
  SHA-256 `057e12b7adc26b11fc4022acedffd17c0bbf786d0111222570398250ef3662be`.
- [Installed wheel run 44812797344](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797344)
  built a PEP 517 wheel and installed it outside the checkout; `pip check`,
  public exports, and `resident-search-v1` passed. Artifact
  [14667767054](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797344/artifacts/14667767054),
  SHA-256 `cadfb46d6191211dbaf420ae3092ae7822ee16b5f4f42120f05dcd5288a7be4f`.
- Rust/permutation-depth CPU contract run
  [44812797266](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797266)
  passed.
- Read-only Vast run
  [44812797614](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797614)
  found instance 61080396 exited and skipped checkout, transfer, build and
  GPU execution. It did not change resource state. Physical 1-/2-GPU runs and
  the штатный checkpoint scorer remain unverified.
- No local build/results were created and no paid execution was used.
- Full receipts: `test_results/refactor_20261120/cnot-collective-preflight/GITHUB_STATUS.md`.

## 2026-11-21 - Prepare installed-wheel GPU acceptance

- Added an installed-wheel permutation acceptance script for CayleyPy LRX(4).
  It searches a reachable state at exact distance 2–4, independently replays
  the result, and repeats each one-GPU and two-GPU search inside a Rust-owned
  session while checking worker-process reuse.
- Converted the existing Vast CNOT control workflow to manual dispatch. It
  requires a USD spend cap, checks the already-running 2x RTX 3060 instance,
  derives a bounded test runtime, and never starts or stops the instance.
- The physical workflow builds and installs the PEP 517 wheel on Vast, checks
  the private trained scorer's pinned checkpoint, runs permutation and CNOT
  searches on one and two GPUs, and keeps full output in the remote
  test_results/ tree and Actions artifacts. Private checkpoint files are
  referenced in place and are excluded from the source archive.
- The source harness has not yet run on physical GPUs. A configured workflow
  and CPU-side syntax validation do not establish CUDA, NCCL, checkpoint
  scoring, or 1-/2-GPU acceptance. A user-supplied spend cap is still required
  before dispatch; the target Vast instance was previously reported stopped.

## 2026-11-21 - Repair GPU acceptance packaging checks

- Included the repository-root `tests/` directory in the remote build archive because the full CMake project references those C++ test sources during configuration.
- The acceptance environment now force-reinstalls the wheel built for the tested commit and verifies that `multigpubeamsearch` imports from that run's virtual environment.
- Kept the authenticated SSH control-master connection open until remote `test_results/` retrieval, even though the one-time key is revoked by the remote exit trap. Hosted fake-SSH tests cover both successful runs and failed runs with partial-result collection.
- The permutation fixture independently replays moves and checks the reported device layout against the count of distinct Rust rank-worker processes; recorded world size now reflects those observed workers.
- Hosted source validation passed in [Actions run 44831036050](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44831036050); artifact [14675264978](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44831036050/artifacts/14675264978), SHA-256 `66fb835caa5cc9e1a897bb97777c18cccdffd9cb3092c3ec2e7908ae9181bfad`.
- That push-triggered run validated Python/Bash syntax and packaging guards only. The physical GPU job was skipped; CUDA/NCCL search and checkpoint-scored CNOT remain unverified pending a supplied spend cap and an already-running instance.
- Source-controlled evidence: `test_results/refactor_20261121/gpu-acceptance/GITHUB_STATUS.md`.

## 2026-11-21 - Require strict CNOT backends and coordinated preparation

- CNOT strict acceptance now compares activation across every rank before
  transport selection. It fails collectively on absent or inconsistent
  DeviceDomain activation, ABI/capability mismatch, wrong scorer/checkpoint/
  manifest/library hashes, and unsupported Stream 5 ABI. The ordinary
  compatibility fallback remains available outside strict acceptance.
- The Rust TQP scorer reports its binary and search-manifest SHA-256 alongside
  the checkpoint metadata. DeviceDomain and Stream 5 dispatch events identify
  the actual implementation, ABI, library hash, rank, world, and device after
  the corresponding operation succeeds.
- CNOT dtype validation uses canonical Torch dtype identity for state/index
  int64 and move int64/uint16 inputs. Winner packet construction and history
  edge allocation/copy are coordinated before exchange/broadcast, keeping the
  live Gloo group usable after rank-local failures.
- Vast source validation now covers private preflight receipt parsing and
  manifest/checkpoint propagation. The explicit GPU workflow accepts only a
  validated 2x desktop RTX 3060 ID and combines wheel/model/backend/lifecycle,
  elapsed-time, and estimated-cost receipts. It does not rent hardware.
- Initial hosted run for `ce74de3` exposed formatting and compile blockers before
  the installed-wheel tests: `cargo fmt --check`, two missing `int64` bindings,
  and an invalid PyO3 extraction of Python `None` into Rust `()`. The next
  revision fixes these source issues; dtype, admission, and Gloo tests remain
  pending until the wheel builds.
- No local build, test, logs, result archives, paid compute, CUDA, or NCCL runs
  were produced. Full hosted logs remain in Actions artifacts.
- Follow-up hosted runs at `01443ef` confirmed the wheel build, installed-wheel
  smoke test, and dtype contract, then exposed two test-harness regressions:
  the admission script executed its multiprocessing body during spawn imports,
  and CPU mocks did not model the new strict-mode agreement votes. The CPU job
  also showed that single-rank compatibility fixtures need no distributed
  agreement method. These are corrected in the next source revision; hosted
  reruns are pending. The physical 2x RTX 3060 gate remains unrun.
- At `d6e1bc2`, hosted CPU/Gloo validation passed Rust checks, PyO3 build, 306
  compatibility tests (2 skipped), and NCCL-probe syntax. Installed-wheel
  build, smoke, and warning-as-error dtype tests passed, then its backend
  admission run found a real variable-length collective contract: different
  strict activation values produced differently sized Gloo tensors. The
  admission contract is now SHA-256-normalized before collective comparison;
  hosted rerun passed the ABI/capability mismatch probes and the normalized
  contract activation mismatch. It then exposed a test assertion expecting
  RuntimeError where the public Python setting-consensus API correctly raises
  ValueError. The assertion is corrected; installed-wheel admission and the
  phase-boundary script then exposed the same multiprocessing main-guard issue;
  it is corrected. Actionlint identified malformed YAML at the skipped-run
  summary line in the new-host workflow; that command is now a block scalar.
  Final hosted admission/phase-boundary/YAML lint checks are pending. Full logs
  remain in Actions artifacts; no local tests or GPU run were produced.
- Follow-up review of the hosted phase-boundary RED showed the fixture used a
  one-gate state that returned before winner redistribution and history-edge
  reconstruction. The fixture now uses a two-gate state and checks recovery by
  replaying the returned sequence. Removed an unused `known_hosts` assignment
  from SSH key attachment after Actionlint flagged it. Hosted rerun is pending;
  no local tests or GPU run were produced.
- The rerun passed installed-wheel build/smoke, dtype warning-as-error, strict
  backend admission, two-rank Gloo failure injection with same-group recovery,
  and public CNOT search. Actionlint then found two unused retry-loop counters
  and an unused remote result-directory variable; they are removed before the
  next lint run. No physical GPU test was run.
- At `78087ab`, the hosted installed-wheel workflow passed all gates including
  Actionlint; the hosted source-validation workflow also passed its instance,
  private-input-receipt, fake-SSH retrieval, and embedded-script fixtures. The
  evidence is CPU/Gloo and harness-only. No physical CUDA/NCCL or trained
  checkpoint search ran. Receipts are in
  `test_results/refactor_20261121/rust-cnot-admission/GITHUB_STATUS.md` and
  Actions artifacts 14701749750 and 14701789289.

## 2026-11-21 - Bind CNOT acceptance to one invocation

- Replaced loose strict-backend marker parsing with a versioned JSON event
  schema shared by Rust and Python. Per-rank JSONL receipts, trace events, the
  result report, source SHA, installed-wheel SHA, process ID and physical GPU
  UUID are checked as one `(run, case)` invocation. Admission ABI and library
  hashes are matched against actual DeviceDomain, TQP scorer and Stream 5
  dispatches; one-rank runs explicitly state that no exchange occurred.
- Added hosted CPU negative fixtures for missing ranks and operations,
  duplicate admission/terminal events, foreign invocation identity, invalid
  ABI/hash, unsupported world size, duplicate GPU UUID and report/trace/receipt
  disagreement. Rust and Python producer serialization are compared with the
  same fixture consumed by the Python verifier.
- `verify_cnot.main` now enters its diagnostic `World` before checked scorer
  setup and leaves it before destroying the process group. The installed-wheel
  workflow invokes that entry under two Gloo ranks, injects a scorer-load
  failure on each rank in turn, and follows each failure with a healthy search
  whose written result is checked. The lifecycle job has a bounded timeout.
- The unpaired duplicate two-GPU trained-deep case was removed. The matched
  one-/two-GPU pair remains in the manual hardware workflow.
- At `fe520518`, hosted source validation `44899191042` and installed-wheel
  validation `44899191012` passed, including the paired-result verifier,
  `verify_cnot.main` lifecycle, public CNOT search and workflow lint. The GPU
  job was skipped; no CUDA/NCCL or trained-checkpoint execution was established.

## 2026-11-21 - Vote rank-local backend preparation

- Added an installed-wheel two-rank Gloo regression scenario for DeviceDomain
  preparation failures after local ABI admission. It injects one failure on
  rank 0 and then rank 1, requires a common phase error, and makes a healthy
  admission/collective on the same process group after each failure.
- Extended the phase-boundary test with failures immediately before the local
  DeviceDomain dispatch trace on each rank, followed by a bounded same-group
  public search and replay. This is intended to catch nested Gloo votes inside
  a local trace callback.
- The test currently references the planned `_probe_cnot_device_domain_vote`
  and `_probe_cnot_device_stage_vote` boundaries and is awaiting hosted RED
  evidence before the runtime changes. No GPU execution has occurred.

## 2026-11-21 - Execute the remote Vast payload on hosted CPU

- Added a fixture that extracts and executes the exact SSH remote payload with
  a cleared environment and a stub acceptance worker. It checks that the
  receipt exists before the worker starts, success and controlled failure
  preserve their status, and the completed log and receipt share the supplied
  `run_root`. Runs `44902996083` and `44903453687` stopped in fixture setup
  (first missing `sys`, then `bash` absent from the intentionally clean PATH)
  before executing the payload. Other hosted CPU verifiers passed; the GPU
  jobs were skipped. Correct the launcher to use `/bin/bash` before counting
  the next run as payload evidence.

## 2026-11-21 - Derive remote results from the transport contract

- Hosted RED `44903569577` executed the extracted remote payload with an empty
  environment and failed before receipt creation on the unbound runner-only
  `REMOTE_ROOT` variable. Fake SSH result retrieval passed on both success and
  failure.
- Split the GPU acceptance body into the archived
  `tools/vast_cnot_gpu_acceptance.sh`; the SSH wrapper now creates one
  `result_dir` from positional `run_root` and the same validated relative path
  passed by the transport helper for result retrieval. It seeds a receipt
  before archive verification/build and finalizes it on every shell exit,
  including controlled errors and timeout signals. The terminal receipt
  records worker and `tee` exit codes. CNOT event validation reads closed
  per-case logs instead of the concurrently written outer tee log. Hosted
  GREEN and worker syntax checks are pending; no GPU run was made.

## 2026-11-21 - Bind paired CNOT verdicts to report digests

- The pair verifier now validates each report's world against its expected
  topology and acceptance envelope, recomputes the canonical report digest,
  and requires valid root and per-layer survivor SHA-256 values for qhat and
  paired trained cases. The per-case backend verdict carries that digest, and
  campaign aggregation requires it to match the result report.
- Hosted RED `44909951209` reproduced five failures: a mutated two-rank report
  could be accepted as world size one; missing digest/root/survivor hashes were
  accepted; and the backend verdict dropped the report digest. The hosted CPU
  source harness `44910114647` and installed-wheel CPU validation
  `44910114686` passed. The GPU job was skipped; no CUDA/NCCL run was started.

## 2026-11-21 - Record the CUDA worker's observed GPU identity

- Strict CNOT acceptance now selects devices by full UUID and asks the CUDA
  driver for the UUID belonging to the worker's current context. Python and
  Rust receipts compare that observed value with the expected rank mapping;
  Gloo carries rank metadata. A CPU fake-driver fixture covers reordered
  visible ordinals, selected-device mismatch, and duplicate expected UUIDs.
- Hosted source harness `44911121921`, installed-wheel check `44911121957`,
  and CNOT CPU compatibility run `44911121931` passed. The GPU job was
  skipped; no physical GPU identity has been observed.

## 2026-11-21 - Bound each Vast phase by the billing deadline

- Vast SSH discovery now selects a complete direct or mapped host/port pair;
  it does not combine fields from different API routes. The absolute deadline
  starts at the API-reported instance start time and deducts the $10 cap using
  the observed hourly rate.
- Private preflight, source staging, upload, remote build/search and retrieval
  each recompute their available timeout from that same deadline. Every phase
  leaves the reserved stop time intact; remote execution also preserves the
  result retrieval window. Private SSH retries share one phase deadline.
- Hosted RED `44912127310` confirmed that the previous transport retrieved
  results despite a synthetic deadline failure. Hosted RED `44912845491`
  exposed a one-second Decimal undercount at a repeating rate; exact rational
  arithmetic now computes the deadline. Hosted CPU harness `44913195522`
  passed the endpoint fixtures, budget phases, remote-payload execution and
  success/failure retrieval. Its installed-wheel/GPU job was skipped. No Vast
  dispatch or physical GPU run was started.

## 2026-11-21 - Guard vendored native build inputs before Vast dispatch

- Hosted RED `44914169925` exposed the missing packager; installed-wheel RED
  `44914169994` showed both vendored headers left the fingerprint unchanged.
- Added one source contract shared by Vast archive creation, wheel bundle
  validation, and native fingerprinting. It includes the vendor tree and
  rejects missing required files even when the generated bundle manifest omits
  them. Hosted source harness `44916537646` passed real archive creation,
  extraction and missing-header cases. Installed-wheel run `44916537642`
  passed both fingerprint changes and bundle rejection when a required header
  is omitted from the generated manifest. Its GPU job was skipped; no paid run
  started.
- The remote preflight checks CayleyPy before long builds; the worker validates
  NCCL header/library and resolves the pinned, SHA-verified CUTLASS cache before
  GPU compilation, passing that explicit path into the permutation smoke.
  Hosted verification is pending.


## 2026-11-22 - Redact CNOT hosted acceptance reporting and validate GPU identity

- The CNOT acceptance and emergency-watchdog workflows no longer place target IDs or spend caps in dispatch inputs or run titles, print private preflight/remote/cleanup output, or retrieve private result archives. They publish only a timestamp and allowlisted status values. The installed-wheel CPU workflow now directs verbose tool output to runner-temporary files and publishes only the same status receipt.
- Added regressions for status-only artifacts, watchdog output, private result transport, and the active CUDA-context identity contract. Fixed the GPU UUID end anchor that rejected valid physical UUIDs; test fixtures now include the complete typed context observation.
- Hosted source harness run [44954310185](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44954310185) and installed-wheel CPU/Gloo run [44954310147](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44954310147) passed at source SHA `a4994a5a394f7d7f370b7ddc04eb08e175fc8367`. Their only uploaded artifacts were redacted status JSON files (310 and 309 bytes). The push-triggered physical CUDA/NCCL job was skipped; no GPU result is claimed.


## 2026-11-22 - Проверять Slurm global/local rank mapping

- Hosted RED `44959263103` подтвердил, что supervisor принимал `rank=11, node=1, local=4` для раскладки `8,6`; ожидаемый глобальный rank — 12.
- Rust runtime и Python compatibility path теперь отклоняют Slurm окружение, если global rank не равен сумме task counts предшествующих узлов плюс local rank. Torchrun, Slurm корректной раскладки, запуск дочернего процесса и транспорт не менялись.
- На SHA `901a7c630d6be25555fe7a8154bd9bb45b537f33` Rust/Python hosted CPU suite [44959505649](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44959505649) прошёл; Python набор: 323 passed, 2 skipped. Installed-wheel suite [44959505809](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44959505809) прошёл. GPU acceptance [44959505792](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44959505792) выполнил CPU harness; физический шаг пропущен.


## 2026-11-22 - Bind paired CNOT acceptance to saturated permutation report

- Replaced the permutation world-size-only check in the paired CNOT verifier with validation of the LRX(4) Rust smoke and saturated LRX(5) B=3 1/2-GPU cases. The receipt now carries the canonical permutation report SHA-256, exact cutoff summary, independent replay status, and per-world rank identities.
- The report includes per-rank candidate/cutoff counts, worker and GPU UUIDs, binary/build digests, two-call Rust-session reuse, full ordered Hash128 survivors, and fixture replay evidence. Search behavior, kernels, scoring and precision are unchanged.


## 2026-11-22 - Validate CNOT Gloo lifecycle receipts

- Keep rank-local engine inputs separate from common Gloo outcomes, with one phase receipt per rank for mismatch, Rust auto retries, and scorer failures on either rank.
- The installed-wheel validator reads exactly the three rank-zero replay reports and checks solved status plus bitwise replay. CPU tests reject missing phases, wrong rank identity, missing report cases, and absent replay.
- This is hosted CPU/Gloo acceptance only; physical CUDA/NCCL remains a separate gate.

- Corrected the installed-wheel workflow to invoke the validator at `tools/cnot_gloo_lifecycle.py`; the previous command referenced a nonexistent file and failed after the Gloo scenarios.

- Pin the public CNOT dispatch decision to one fixed rank-wide capability intersection and execute the selected route without a second Python engine decision.
- Preserve explicit `engine="rust"` capability errors as `NativeUnavailable`; move the mixed-engine Gloo regression through the public `search_cnot` entry and keep compatibility mocks aligned with the selected-route contract.

- Print the synthetic installed-wheel panel endpoint traceback on failure, and the panel handoff unittest log on the producer CPU gate, so CI failures retain the actual assertion rather than only an exit code.

- Keep panel handoff diagnostics specific: distinguish receipt/source revision disagreement, selected checkpoint disagreement and accepted-update mismatch; correct the installed endpoint oracle for the input tensor's batch dimension.


## 2026-11-22 - Reach injected scorer in the public Gloo verifier lifecycle

- The installed two-rank lifecycle test supplies inert `--library` and `--manifest` arguments for trained cases so the public CLI reaches the injected synthetic scorer factory. No model or checkpoint files are opened by this CPU fixture.
- This preserves validation of the real CLI contract while exercising rank-local scorer failure, common phase reporting, and healthy retry under Gloo.


## 2026-11-22 - Rust-owned typed CNOT search request

- Added `beam_runtime::cnot_search::CnotRequest` with typed mode, validation policy, requested engine, width, capped shortlist, depth, and wire count.
- The Rust controller now creates this request inside the existing collective input-admission phase and uses the validated values for native stage execution. Public input packing, scoring, stage order, kernels, and rank collectives are unchanged.
- Runtime unit tests cover preserved controls, alphabet shortlist capping, selected-engine semantics, and rejection of invalid public options.


## 2026-11-22 - Revalidate Gloo lifecycle receipts and panel handoff byte binding

- The installed two-rank `verify_cnot` check records rank-local receipts with `case_id`, `phase` and `common_outcome`; it validates each rank's requested engine separately. It reads exactly three successful bitwise-replay reports written by rank 0: `healthy-after-engine-mismatch-0.json`, `healthy-after-0-0.json`, and `healthy-after-1-0.json`.
- The panel-v2 resolver now hashes and parses each pinned JSON artifact from the same in-memory bytes, including both panel and baseline manifests. RED was reproduced by the one-read-per-artifact regression at run [44983615094](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44983615094); the fix is in code SHA `57bf4f0deb90e1b0009b1a458968e5ca4096fe23`.
- At that SHA the installed PEP 517 wheel passed all 27 steps in [run 44983817156](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44983817156), including the two-rank Gloo lifecycle/receipt verifier and panel endpoint tests. The pinned producer CPU compile and handoff tests passed in [runs 44983817105](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44983817105) and [44983819836](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44983819836).
- Physical CUDA/NCCL was skipped. These results establish hosted CPU, Gloo, and installed-wheel behavior only.

## 2026-11-22 - Installed wheel validation for public cluster workers

- Added an installed-wheel check for the packaged scheduler worker and Rust rank supervisor, run from outside the checkout with `PYTHONPATH` unset and `REQUIRE_RUST_CLUSTER=1`.
- Exercised Python and Rust supervisors, both torchrun local-rank flag spellings, shared rank environment, Slurm mapping, peer failure, timeout, and SIGTERM child cleanup. The Python default remains unchanged pending physical parity.
- A first hosted harness run rejected the proposed raw-log artifact because the installed workflow must publish only redacted status. Removed the extra log artifact and `tee`; the source harness then passed.
- Installed wheel passed on source SHA `3241de1df83c101abe3cca906fa9d1de801f611a` in [run 44986148911](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44986148911). Hosted acceptance harness passed in [run 44986148934](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44986148934); panel-v2 CPU compile/handoff passed in [run 44986152290](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44986152290). Physical GPU job skipped.
