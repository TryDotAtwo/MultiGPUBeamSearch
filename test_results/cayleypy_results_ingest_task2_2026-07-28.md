# CayleyPy Results Ingest Task 2 round-2 gate (2026-07-29)

Private CPU Kaggle kernel `trydotatwo/cayleypy-results-ingest-npm-gate` v12 completed with Node v20.19.0/npm 10.8.2. Lock generation, `npm ci`, 11 Node schema tests, 18 real Miniflare D1/R2 receipt/recovery tests, and TypeScript typecheck all passed.

The v12 tests prove normal concurrent duplicates enqueue once, stale `received` duplicates may resend the same submission id after the bounded wait while retaining one D1 row, crash-before-send and stale `retryable` rows are recovered by the bounded helper, Queue/consumer interleavings accept already-validating work, ambiguous Queue failure is resolved by a checked reread, successful recovery clears `safe_error`, and repeated Queue failure advances `retry_count`/`updated_at` so a page beyond `limit` is not starved. Raw R2 remains retained throughout recovery.

Downloaded `payload-sha256.json` matched all 14 current input files. Key SHA-256 values are `migrations/0001_initial.sql=8c3fe6fdc4381e123a901593962194f90aae7bfc484e6a6f5685195d05cb0ba6`, `src/db.ts=a16b27d6c09808a22a52824f9a1b8997b86e6898c8b6d04463df8c4551378221`, `src/storage.ts=5727b582b69318766f07c60f859a7603097922572df9bfa8723ffb5e3f8e6553`, and `test/receipt.test.ts=c6215277858f55d8422e23244254a4761b3ba802efde4b99f7fc8ac645120698`. Pulled metadata confirmed `is_private=true` and `enable_gpu=false`. Evidence is under `test_results/kaggle_cayleypy_results_ingest_npm_gate/outputs_v12/`.

v11 was a diagnostic hang: it remained RUNNING through the live-output check at 2026-07-29 00:55:20 +03:00 and yielded no downloadable files. The cause was a test-only `await Promise.resolve()` polling loop that could starve runtime timers; production code was unaffected. Deferred Queue-start signals replaced the loop before exact v12.

Miniflare reports a fallback from configured compatibility date 2026-07-28 to 2025-07-12; this is recorded and not suppressed.
