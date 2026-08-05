# CayleyPy Results Ingest Task 2 round-4 verification (2026-07-29)

## Round-4 delivered

- Added the reusable current `parkPausedSubmission(db, submissionId)` storage primitive without implementing the Task 3 Worker or Task 4 Queue handler. It conditionally transitions `received|queued|retryable -> retryable`, writes only `safe_error=ingest_paused`, leaves `retry_count` and raw R2 untouched, refreshes `updated_at`, rereads D1, and returns only after the durable state is verified. A false transition is accepted only when that reread already proves the same park.
- Extended bounded normal recovery to stale `queued` alongside `received|retryable`. A successful resend of the same `{submission_id}` performs `queued -> queued`, refreshes `updated_at`, clears `ingest_paused` or another stale `safe_error`, retains raw, and therefore leaves the row ineligible for the next old-cutoff cron page. Queue-send failure from queued remains recoverable as retryable and advances retry metadata.
- Added real-migration Miniflare tests for all three park source states, unchanged retry budget/raw, 101 duplicate parks with one D1 row, transition-false+reread, repeated D1 park errors, and same-id stale-queued normal recovery. The full receipt suite grew from 19 to 23 tests.
- Corrected the future Task 4 contract: resolve mode first; in non-normal modes parse only `submission_id`; durably park and then ACK; never call `message.retry()` after successful parking; and touch no R2/replay/publisher/token/GitHub/payload/raw-mode-log surface. D1 park failure remains on the platform retry/exhaustion path, with stale queued normal recovery as the loss-prevention backstop.
- Corrected the future GitHub writer contract: persist validated ids durably before the final mode guard, re-arm a bounded alarm outside normal, and throw/retry if `setAlarm` fails. No Task 3/4 handler, fourth mode, Cloudflare resource, deployment, secret, GitHub mutation, or public publication was added.

## TDD RED: private CPU v15

- Kernel: `trydotatwo/cayleypy-results-ingest-npm-gate`, immutable version 15.
- Terminal notebook status: `KernelWorkerStatus.COMPLETE`; the embedded gate correctly records RED because command results are captured rather than raised as a notebook exception.
- Pulled metadata: expected id, `is_private=true`, `enable_gpu=false`, `enable_tpu=false`, `machine_shape=None`, `enable_internet=true`.
- Node `v20.19.0`; npm `10.8.2`; lock generation and `npm ci` passed.
- `npm test`: RED. Schema was 11/11 green; the existing receipt suite was 19/19 green; exactly four new tests failed with `expected undefined to be type of 'function'` because `parkPausedSubmission` did not exist. TypeScript typecheck passed, `npm-test` exited 1, and `all_commands_passed=false`.
- Exact RED payload anchors: `src/db.ts=a16b27d6c09808a22a52824f9a1b8997b86e6898c8b6d04463df8c4551378221`, `src/storage.ts=5727b582b69318766f07c60f859a7603097922572df9bfa8723ffb5e3f8e6553`, `test/receipt.test.ts=f1428d528a226d70ec7304e7b67e678021d2c03fb189e5a80e98c009c923fad9`.

Preserved v15 evidence SHA-256:

- `outputs_v15/payload-sha256.json`: `aaec5cb3418b0070cfbf2f143f806eb4facdc3bced2fffcd0d490836b1b45442`
- `outputs_v15/npm-test.log`: `73c53a298c8b58cc16104d5f4cc95602bf5ade66f605a03fa6b474a01da162a7`
- `outputs_v15/npm-gate-results.json`: `2e0e7fad28ae683ff85535a11a6aa2cdfd8ebf081507685ec7a48622eced77ba`
- `pulled_v15/kernel-metadata.json`: `572d93ed4dd74d3aeabba1951ab4e098cbf6d217219c5b9edddab17bdbb35ae0`
- `pulled_v15/cayleypy-results-ingest-npm-gate.ipynb`: `901795f45e754e145ff5aacfdde788fc338ba3d84058544cc4e77f6af02279a3`

## GREEN: private CPU v16

- Kernel: `trydotatwo/cayleypy-results-ingest-npm-gate`, immutable version 16; terminal status `KernelWorkerStatus.COMPLETE`.
- Pulled metadata again proves private CPU execution with Internet enabled and no GPU/TPU/machine shape.
- All six gate commands exited 0: Node/npm version checks, lock-only install, `npm ci`, `npm test`, and `npm run typecheck`; `all_commands_passed=true`.
- `npm test`: PASS - 11/11 schema tests and 23/23 real Miniflare migration/D1/R2 receipt/recovery tests. The 101-duplicate park case itself took 1.620 seconds. The pinned runtime still emits the recorded compatibility fallback warning from requested `2026-07-28` to supported `2025-07-12`.
- Exact downloaded `payload-sha256.json` was rechecked against all 15 current worktree inputs: 15 checked, zero mismatches.

Key v16 embedded payload SHA-256 values:

- `migrations/0001_initial.sql`: `8c3fe6fdc4381e123a901593962194f90aae7bfc484e6a6f5685195d05cb0ba6`
- `src/db.ts`: `d46481ee7eb8a7d2cbc3673f305566b31b6605eb4c6cae1d7547cfbec72a6609`
- `src/storage.ts`: `33bd45217acef630ed8bb54b70facd9b69776b6a8c5bfb34721d4e79d9dd14bf`
- `test/receipt.test.ts`: `686b3beb31adb722f1647bbd8a9ab95807edb06419222178cfe436eb1b2e6094`

Preserved v16 evidence SHA-256:

- `outputs_v16/payload-sha256.json`: `4ed67212b37f336527f475f212cc1ea586bc49bf91f0b0184c2151e25facf91f`
- `outputs_v16/npm-test.log`: `e7d4b0cf734a14a9f598975096667c5760aeef9a57c0bece7bd2c0ac920bd34a`
- `outputs_v16/npm-gate-results.json`: `7f0bc2d34a6c521dade186bc3076f0a0cc6a1fd3676806b747cb322284130132`
- `pulled_v16/kernel-metadata.json`: `572d93ed4dd74d3aeabba1951ab4e098cbf6d217219c5b9edddab17bdbb35ae0`
- `pulled_v16/cayleypy-results-ingest-npm-gate.ipynb`: `29ab7adc4e402752b5c0756807a5a8814b3c55d352649e539bbe16b5eb0f9107`

## Prior round-3 TDD and v13 failure evidence

The migration contract was first exercised against the old handwritten receipt schema and failed because it lacked the deployable `state` CHECK and `submissions_recovery` index. Exact private Kaggle v13 then provided the runtime RED case for the first raw-loader implementation: 11/11 schema tests passed and TypeScript typecheck passed, but `env.RESULTS_DB.exec()` treated the multiline `CREATE TABLE` as incomplete, so the receipt suite failed in setup and all 19 receipt tests were skipped. `npm test` exited 1 and `all_commands_passed=false`.

Kaggle versions are immutable, so v13 could not be corrected in place. Its failure is preserved under `test_results/kaggle_cayleypy_results_ingest_npm_gate/outputs_v13/`; `npm-test.log` SHA-256 is `d24646c0ac44479a39e203777834e72bceaf3ec08ef2f32fbb0c0e53a8999228` and `npm-gate-results.json` SHA-256 is `8150d39418920225250c6fa57d6cd058426cf4c45982831f56ad6264295549df`.

The fix follows the exact APIs exported by pinned `@cloudflare/vitest-pool-workers=0.8.55`: Node-side `readD1Migrations()` delegates SQL splitting to Wrangler, and Worker-side `applyD1Migrations()` applies and records the parsed migration. This avoids both raw multiline `D1.exec()` and a second handwritten schema.

## Prior private CPU v14 runtime gate

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

No Task 3 Worker source, Task 4 consumer/replay source, GitHub writer implementation, Cloudflare deployment/resource/secret, git push, GitHub mutation, or public publication was created. The exact private v16 gate is the dependency/runtime verification; v15 preserves the behavior-first RED evidence. The pinned Vitest pool still warns that its newest emulated compatibility date is `2025-07-12` and falls back from configured `2026-07-28`; this limitation is recorded and no dependency upgrade was made. Kaggle's Windows CLI downloaded every requested artifact before returning exit 1 on console encoding of a checkmark; the JSON results and files above are complete and were inspected directly.
