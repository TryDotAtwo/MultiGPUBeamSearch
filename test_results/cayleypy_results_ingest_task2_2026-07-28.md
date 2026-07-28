# CayleyPy Results Ingest Task 2 round-3 verification (2026-07-29)

## Delivered

- Replaced the receipt suite's handwritten D1 schema with the pinned Cloudflare migration harness. `vitest.config.ts` calls `readD1Migrations()` on the real `migrations/` directory, injects the parsed migrations through `TEST_MIGRATIONS`, and `test/apply-migrations.ts` calls `applyD1Migrations()` against Miniflare `RESULTS_DB`.
- Added a real-migration regression test that proves `0001_initial.sql` was recorded in `d1_migrations`, `submissions_recovery` has the exact `(state,updated_at)` columns, and the deployable `state` CHECK rejects an invalid state.
- Tightened the future operating contract without implementing Task 3. `INGEST_MODE` is the exact case-sensitive allowlist `normal|store_only|reject`; missing, empty, mixed-case, and unknown values fail closed as `reject`. Normal HTTP persists and queues, `store_only` persists raw R2 plus D1 `received` without Queue publication, and reject/fail-closed modes accept and persist nothing.
- The scheduled recovery contract is a strict no-op outside `normal`; a stale `store_only` row remains unqueued until normal mode resumes. The future Task 4 Queue handler parks every non-normal backlog message for a fixed bounded 300 seconds before any state transition, R2/replay validation, or publication enqueue, without payload/raw-mode logs. The future GitHub writer rechecks mode immediately before external authentication or GitHub mutation and retains validated ids without `staged|published` transitions outside `normal`.
- Added explicit future tests for all modes, stale `store_only` recovery, non-normal Queue backlog isolation, the GitHub final guard, and normal-mode resumption. No fourth operating mode was introduced.

## TDD and v13 failure evidence

The migration contract was first exercised against the old handwritten receipt schema and failed because it lacked the deployable `state` CHECK and `submissions_recovery` index. Exact private Kaggle v13 then provided the runtime RED case for the first raw-loader implementation: 11/11 schema tests passed and TypeScript typecheck passed, but `env.RESULTS_DB.exec()` treated the multiline `CREATE TABLE` as incomplete, so the receipt suite failed in setup and all 19 receipt tests were skipped. `npm test` exited 1 and `all_commands_passed=false`.

Kaggle versions are immutable, so v13 could not be corrected in place. Its failure is preserved under `test_results/kaggle_cayleypy_results_ingest_npm_gate/outputs_v13/`; `npm-test.log` SHA-256 is `d24646c0ac44479a39e203777834e72bceaf3ec08ef2f32fbb0c0e53a8999228` and `npm-gate-results.json` SHA-256 is `8150d39418920225250c6fa57d6cd058426cf4c45982831f56ad6264295549df`.

The fix follows the exact APIs exported by pinned `@cloudflare/vitest-pool-workers=0.8.55`: Node-side `readD1Migrations()` delegates SQL splitting to Wrangler, and Worker-side `applyD1Migrations()` applies and records the parsed migration. This avoids both raw multiline `D1.exec()` and a second handwritten schema.

## Private CPU v14 runtime gate

- Kernel: `trydotatwo/cayleypy-results-ingest-npm-gate`, version 14.
- Terminal status: `KernelWorkerStatus.COMPLETE`.
- Node `v20.19.0`; npm `10.8.2`.
- Pulled metadata: `is_private=true`, `enable_gpu=false`, `enable_tpu=false`, `machine_shape=None`, expected kernel id.
- `npm install --package-lock-only --no-audit --no-fund`: PASS.
- `npm ci --no-audit --no-fund`: PASS.
- `npm test`: PASS - 11/11 schema tests and 19/19 real Miniflare D1/R2 receipt/recovery/migration tests. The receipt suite reports 262 ms of setup time for migration application.
- `npm run typecheck`: PASS.
- `npm-gate-results.json`: all six commands exit 0 and `all_commands_passed=true`.
- Downloaded `payload-sha256.json` matches all 15 current worktree inputs byte-for-byte.

Key embedded payload SHA-256 values:

- `migrations/0001_initial.sql`: `8c3fe6fdc4381e123a901593962194f90aae7bfc484e6a6f5685195d05cb0ba6`
- `vitest.config.ts`: `2fe870a0415122cf35ed9d4c29d678408f356174b263205f01cea39855d6fa7e`
- `test/apply-migrations.ts`: `f7cc52a868ed3bf5ae2cf93fead32335a1541019d7828e16eb1375b0ebe5918d`
- `test/receipt.test.ts`: `2130b66464e60137c5ee4051677077e3caa85f761573b39ddde16542a0798f21`
- `src/db.ts`: `a16b27d6c09808a22a52824f9a1b8997b86e6898c8b6d04463df8c4551378221`
- `src/storage.ts`: `5727b582b69318766f07c60f859a7603097922572df9bfa8723ffb5e3f8e6553`

Evidence paths and artifact SHA-256 values:

- `test_results/kaggle_cayleypy_results_ingest_npm_gate/outputs_v14/payload-sha256.json`: `07a34f3ae74b2500f4ac97dd4391dda4986da6f6706492baf9bc0ce4433572b4`
- `test_results/kaggle_cayleypy_results_ingest_npm_gate/outputs_v14/npm-test.log`: `2fbbaa3bc351f493f78bf1da6a06136b57322fbe4d3101e3f7aca31b0576d0d8`
- `test_results/kaggle_cayleypy_results_ingest_npm_gate/outputs_v14/npm-gate-results.json`: `ed936a5d5ef2435900fe21fc8245d66733f8dad17a227b0d5a3ff151fc34da4c`
- `test_results/kaggle_cayleypy_results_ingest_npm_gate/pulled_v14/kernel-metadata.json`: `572d93ed4dd74d3aeabba1951ab4e098cbf6d217219c5b9edddab17bdbb35ae0`
- `test_results/kaggle_cayleypy_results_ingest_npm_gate/pulled_v14/cayleypy-results-ingest-npm-gate.ipynb`: `3a2d985de71a5f70f49ecc755fd18d2c3e0c7a8b4fdedbfe075694d1781ee249`

The downloaded regenerated lockfile is text-equivalent to the embedded lockfile; npm normalized CRLF to LF, so its downloaded byte hash is `fec430c2850c985fb1bd3404ca0fbc85c4b50398e795ce25e4eb865f0ce19064` while the exact embedded input hash remains `6c84152419596a537b4f0df8dee1ab2e843103faf606506bd46a6483fcf10b81`.

## Boundaries and runtime note

No Task 3 Worker source, Cloudflare deployment/resource/secret, git push, or public publication was created. The failed local npm installs left only a partial `node_modules` tree, which was removed before staging; the exact private v14 gate is the dependency/runtime verification. The pinned Vitest pool still warns that its newest emulated compatibility date is `2025-07-12` and falls back from configured `2026-07-28`; this limitation is recorded and no dependency upgrade was made.
