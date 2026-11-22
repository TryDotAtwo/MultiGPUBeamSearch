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