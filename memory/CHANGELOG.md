## 2026-11-23 - Supervise local permutation ranks from Rust

- Routed the public `local_launcher.run_local_ranks` through the installed Rust runtime. Rust owns local process startup, the existing rank arguments/environment and log paths, timeout/cancellation polling, and process-group cleanup; older wheels retain the Python compatibility supervisor.
- Added installed-wheel coverage from outside the checkout for two synthetic ranks and sibling cleanup after a rank failure. Search stages, kernels, NCCL, ordering and numerical policy are unchanged.
- Installed-wheel validation passed on source SHA `fd42ebca8d7486c92a7d3a26c98a7ea591cc6f8f`: [run 45131297173](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45131297173). Physical permutation CUDA/NCCL was not run.

## 2026-11-22 - Resolve permutation solution policy in Rust

- Added a versioned `beam-runtime` request for first/collect mode, extra collection depths, the collected-solution limit, and the result TSV path.
- Routed the public permutation runner environment through the installed Rust resolver after calibration. Preserved identical string values and the Python fallback for older wheels.
- Added installed-wheel coverage for collection and first-solution policy, invalid mode rejection, and compatibility fallback. No search, kernel, precision, or transport behavior changed.
- Installed-wheel validation and saturated permutation CPU acceptance passed on source SHA `c9b6e10b2f013d7fc243719d1d7316faba78b88b`: [run 45062575238](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45062575238). Physical CUDA/NCCL was not run.

## 2026-11-22 - Build fixed permutation launch environment in Rust

- Moved the public runner's puzzle, input, weight, history, and base BFS environment variables into a versioned `beam-runtime` function.
- Kept model-specific inference variables in the existing adapter and preserved the Python environment builder for older wheels.
- Added an installed-wheel regression for the actual Rust function through the public permutation environment helper and for the compatibility fallback.

## 2026-11-22 - Resolve permutation CUDA availability and device selection in Rust

- Added a versioned runtime plan that rejects unavailable CUDA before selecting process-local device indices, preserving existing explicit, `num_gpus`, graph-placement, and error rules.
- Routed the public permutation preparation helper through that plan. Kept Linux/torchrun checks and old-wheel compatibility in place.
- Added installed-wheel coverage for the real public Rust plan, compatibility fallback, and CUDA-unavailable short-circuit.
- Installed-wheel and saturated permutation CPU acceptance passed on source SHA `e4e96c452dba8b531c7de262f8f333b088a579c3` ([run 45058727951](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45058727951)); the shared Rust CPU/Gloo check passed ([run 45058728166](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45058728166)). Physical CUDA/NCCL was not run.

## 2026-11-22 - Test child CUDA mapping at its public boundary

- Kept the installed-wheel CUDA visibility test focused on the direct mapping helper now that the public child launcher uses the combined Rust child-environment plan.
- The end-to-end child environment test continues to verify the installed Rust resolver, filtering, selected CUDA mask, NCCL path, legacy fallback and error behavior.
- The current migration phase prioritizes permutation search; CNOT code and compatibility remain preserved while CNOT-specific development and acceptance are deferred.
- Installed-wheel validation passed on source SHA `1e8b88cf6f188171bbe583875a400fca9095c5f8` in [run 45056737762](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45056737762), including direct CUDA mapping, the public combined child-environment resolver, and saturated permutation CPU cases.

## 2026-11-22 - Move public cluster planning into beam-runtime

- The public multi-node `plan_cluster` adapter now delegates block rank placement, exact beam capacity and aligned shard geometry to a typed Rust plan.
- Kept the same Python API, validation limits and result dataclasses. Wheels without the Rust planning function keep the existing Python formula.
- Added installed-wheel assertions for the real public helper and legacy fallback, plus exact Rust CPU checks for uneven node sizes and overflow rejection.

## 2026-11-22 - Move permutation microbatch planning into beam-runtime

- Added a typed Rust request and plan for the existing integer microbatch formula; kept all alignment, capacity, row-limit, scalar-head, ensemble and environment semantics unchanged.
- The public Python helper now consumes the Rust plan and retains the old Python formula only when the installed extension lacks the API.
- Added installed-wheel assertions for the exact public helper output and old-wheel fallback.

## 2026-11-22 - Keep CNOT history edges typed across checked stages

- Generalized the Rust checked-stage boundary to coordinate a Rust-owned result value directly, while preserving the existing Gloo vote, diagnostics and cancellation behavior.
- Decoded the fixed history wire row into the runtime Edge type inside the same checked phase. This removes the Rust-to-Python-tuple-to-Rust roundtrip and keeps malformed-row failures on the common stage vote.
- Added installed-wheel Gloo cases that corrupt one rank's history row after broadcast on rank 0 and rank 1 in turn, then verify common error reporting and a healthy retry on the same group.

## 2026-11-22 - Carry one ranked CNOT request into native stages

- The Rust CNOT controller now passes its validated RankedCnotRequest into the native-stage adapter as one private PyO3 value.
- The controller regression asserts that this typed request reaches native-stage dispatch together with the admitted topology.
- The native stage derives rank, world size, mode, audit flag, and Rust search options from that request, then checks its topology against the live world before entering the search.
- The existing scalar extension arguments remain available to direct legacy callers. Tensor layout, scorer, collectives, kernels, and search policy are unchanged.
- Hosted CPU and installed-wheel checks are required for this Rust adapter boundary.

## 2026-11-22 - Run the panel-v2 identity tests in the CPU adapter gate

- The bridge's typed panel-v2 manifest and identity validator now live in a shared Rust module compiled with the CPU Torch feature; the CUDA C ABI bridge imports the same implementation.
- The hosted Rust gate selects the four named manifest tests from that CPU module and checks that each actually ran. This avoids the previous zero-test filter and avoids unrelated full-crate tests.
- The panel manifest source is included in native preparation, cache fingerprints, and routing classification.

## 2026-11-22 - Require panel-v2 Rust unit tests to execute

- Removed the over-specific Cargo test-name filter that returned success with zero tests (48 filtered out) on the pinned producer crate.
- The adapter gate now runs the library test suite and requires all four panel-v2 scorer manifest cases to appear in successful output before it can save the native build cache. A zero-test or missing-case run fails with a concise marker.
- Producer routing, cache keys, and inference-policy behavior are unchanged.

## 2026-11-22 - Keep installed CNOT dispatch diagnostics redacted

- The two-rank installed-wheel job keeps the complete diagnostic log in the runner's private temporary directory. On success it emits only three fixed PASS markers; on failure it emits only the fixed phase and exception type from the test's structured marker.
- This preserves the workflow's redacted-status contract while making the failing phase visible in public Actions output. It does not change dispatch behavior or test coverage.

## 2026-11-22 - Move the CNOT public-dispatch fingerprint into beam-runtime

- Added a typed Rust digest for the CNOT route request and exposed it through PyO3. The public Python entry uses this API inside the existing checked Gloo phase; older wheels retain the previous byte-for-byte Python fallback.
- Installed-wheel coverage compares the Rust and legacy digests across engine values, strict mode, and the u64 depth boundary, then exercises both the installed Rust API and the older-wheel fallback through public `search_cnot`.
- Search ordering and semantics, stage calls, scorer, CUDA kernels, and collective ordering are unchanged. Physical CUDA/NCCL execution remains a separate acceptance gate.

## 2026-11-22 - Accept the current panel-v2 producer revision

- Added producer source revision `eafc0f82f088603a525f428a8fa3d00e9e702fa8` to the consumer allowlist under the existing emitter ABI hash. Kept support for the prior `42ed72f587ab70fc74a85cf0a25001efb29cb994` and historical `7310b1eb5ec07c76c131ba8c60b105ff4aad7a9c` receipts.
- Updated the hosted no-Torch emitter fixture to load the actual producer serializer at `eafc0f82f088603a525f428a8fa3d00e9e702fa8`, create a receipt, and pass it through the consumer resolver. Added a regression for the previous source pin. The no-Torch fixture passed on commit `d3bb737777ba603a002099f1de95de6b85de9eb4` ([run 45029744326](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45029744326)); installed-wheel validation passed ([run 45029740790](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45029740790)). Physical GPU jobs were skipped; no paid run occurred.

## 2026-11-22 - Include the resolved inference plan in Rust session identity

- The installed-wheel identity test requires the schema-v3 native extension and passes its canonical JSON identity through the real PyO3 `SearchSession` API; the session test checks same-plan reuse and invalidation after plan or model changes.
- Extended typed permutation preparation identity with the resolved backend, policy and row limit from the Rust inference resolver. A repeated `NativeSession` call can reuse its resident rank owner only when the plan and other preparation inputs match; a changed plan invalidates preparation.
- Kept schema-v2 extension compatibility by carrying the plan inside its existing structured options object.
- The installed-wheel inference-policy check calls the actual installed Rust extension through `prepare_runtime` and verifies the runner selection and resulting profile.

## 2026-11-22 - Split panel-v2 producer feedback and build caches

- Hosted routing contracts inspect the actual venv-qualified handoff command and require preparation to precede Rust/PyTorch setup.

## 2026-11-22 - Coordinate compatibility-route CNOT identity setup

- Moved CNOT identity construction in the admitted Rust compatibility route into a rank-wide checked Gloo phase before transport startup. A rank-local allocation/initialization error now returns the same phase and cause on every rank.
- The installed-wheel Gloo test hides the newer controller ABI, injects the identity failure independently on rank 0 and rank 1, confirms transport does not start for the failed request, and retries a public search on the same process group.
- Search rules, packed state representation, scorer, CUDA kernels, precision, and collective transport are unchanged.

## 2026-11-22 - Resolve permutation device selection in beam-runtime

- Moved device-list policy from the Python backend into a typed Rust runtime function. The adapter supplies the visible CUDA count, explicit `devices` or `num_gpus`, and normalized CayleyPy/Torch graph-device observations.
- Preserved explicit selection, exact requested GPU counts, all-visible-GPU selection for absent/CPU graph devices, configured CUDA order, `cuda` mapping to ordinal zero, the 128-device limit, and existing rejection cases.
- Wheels without the new PyO3 function keep the Python policy. Runtime kernels, precision, scoring, graph search, and launch visibility mapping are unchanged.

## 2026-11-22 — Rust scheduler rank resolution

- The public torchrun/Slurm rank parser now uses the typed result from `beam-runtime` when the installed wheel provides it. The Python API and `ScheduledRank` dataclass remain unchanged; older wheels retain the existing Python parser.
- Rust validation errors are mapped back to the public `ValueError` contract. A supported Rust resolver failure cannot silently switch the rank assignment path.
- This transfers scheduler metadata only; it does not change device binding, rank launch selection, NCCL transport, or GPU execution.


## 2026-11-22 - Resolve public scheduler rank in beam-runtime

- Added a typed Rust/PyO3 resolver for torchrun and Slurm scheduler environments and routed the public Python `cluster.scheduler_rank` helper through it.
- Preserved the `ScheduledRank` API, malformed-input `ValueError`, and compatibility fallback for installed wheels without the resolver. Errors from a present Rust resolver do not silently switch rank assignment paths.
- Installed-wheel validation passed on `a96be795a011174d7f244c06f1593a62b16e6c27` ([run 45052314835](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45052314835)); Rust CPU/Gloo and hosted acceptance passed ([45052314834](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45052314834), [45052315268](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45052315268)). Physical CUDA/NCCL was skipped.


## 2026-11-22 - Resolve scheduled CUDA ordinal in Rust

- Added a typed `beam-runtime` rule for mapping the local scheduler rank into the visible CUDA namespace and exposed it through a separately versioned PyO3 API.
- Routed `ScheduledRank.cuda_ordinal` through the installed extension; kept the same Python method, exact validation behavior, and fallback when an older wheel lacks the function.
- Added Rust runtime contract tests and installed-wheel checks for a single visible device, local ordinal mapping, insufficient visibility, error propagation, and legacy fallback. No device binding or NCCL behavior changed.
- Rust CPU/Gloo passed in [run 45054019605](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45054019605), installed-wheel validation passed in [run 45054019597](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45054019597), and the hosted acceptance harness passed in [run 45054020205](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45054020205); the physical GPU job was skipped.


## 2026-11-22 - Resolve child CUDA visibility in beam-runtime

- Moved logical CUDA-index mapping through inherited `CUDA_VISIBLE_DEVICES` into a typed Rust runtime function and exposed it through the existing versioned device-binding PyO3 API.
- The public `child_environment` path now uses the installed resolver; it preserves input order, duplicate indices, token trimming, exact invalid-token errors, and compatibility fallback for older wheels.
- Added a hosted installed-wheel test through child environment construction and Rust runtime contract cases. Device selection, physical binding, CUDA kernels and NCCL setup are unchanged.
- Rust CPU/Gloo, installed-wheel, and hosted source acceptance passed on `db3e1be`: [45055128201](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45055128201), [45055128252](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45055128252), [45055128570](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45055128570). Physical CUDA/NCCL was skipped.


## 2026-11-22 - Build permutation worker child environment in Rust

- Moved reserved-variable filtering, selected CUDA visibility, CUDA device-order preservation and NCCL ID path insertion into a versioned `beam-runtime` operation.
- Routed the public permutation `backend.child_environment` through the installed PyO3 extension; retained the same output keys and Python fallback when the wheel lacks the operation.
- Added installed-wheel and Rust contract coverage for environment isolation, selected device order, inherited device identifiers, invalid mapping and older-wheel fallback. Search, kernels, precision and transport are unchanged.


## 2026-11-23 - Resolve permutation search phases in beam-runtime

- Added a versioned Rust phase plan for the existing permutation inference calibration, capacity discovery, maximum-frontier refinement, downstream calibration and working-capacity selection branches.
- The public runner consumes the Rust plan while preserving stage order, model/backend predicates, delayed capability errors and the existing Python implementations of each stage. Older wheels use the matching compatibility plan; errors from the installed Rust resolver fail explicitly.
- Added Rust plan contract cases and an installed-wheel check through the runner's phase-plan helper. Hosted checks are pending for the introducing commit; no physical CUDA/NCCL run was performed.
