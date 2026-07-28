# CayleyPy Results Ingest Task 2 verification (2026-07-29)

## Delivered

- Added migration `0001_initial.sql` for the submission state machine and indexes.
- Added deterministic canonical JSON and SHA-256 semantic idempotency. Client transport fields (`submission_id`, supplied idempotency key, and timestamp) are excluded from semantic identity.
- Raw payload keys are service generated as `raw/v1/YYYY/MM/DD/<uuidv7>.json`; the client supplies no storage or repository path. R2 uses `If-None-Match: *` and stores SHA-256 custom metadata.
- Receipt order is R2 raw object, D1 `received`, awaited durable Queue write, confirmed D1 `queued`, then receipt. Queue failures retain raw R2 and return retryable only after D1 confirms `retryable` with safe code `queue_unavailable`.
- Concurrent duplicates that observe `received` use a bounded D1 reread loop. They cannot return `queued` before the winner's Queue write resolves. A bounded timeout fails with safe code `duplicate_wait_timeout`.
- An `ON CONFLICT` loser deletes and verifies absence of only its own service-generated raw key before returning the winner receipt. Failed cleanup returns `duplicate_raw_cleanup_failed`; an ambiguous D1 insert exception preserves immutable raw for operator recovery and returns `submission_persist_failed`; it is never destructively cleaned.
- Queue errors and D1 transition errors are handled separately. Every compare-and-transition result is checked; a false result is accepted only when a reread proves the required settled state. Other conflicts fail with safe, value-free codes.

## Private CPU runtime gate

- Kernel: `trydotatwo/cayleypy-results-ingest-npm-gate`, private, CPU-only, Internet enabled, version 10.
- Node `v20.19.0`, npm `10.8.2`.
- `npm install --package-lock-only --no-audit --no-fund`: PASS.
- `npm ci --no-audit --no-fund`: PASS.
- `npm test`: PASS — 11 schema tests in Node plus 15 receipt/concurrency/order/cleanup/retry/transition tests with real Miniflare D1/R2 bindings.
- `npm run typecheck`: PASS.
- Exact embedded/downloaded SHA-256: `src/storage.ts=ecd55cb4088a35b9672314e149d44bff83c5af69316dff3d87d78581d1e0a2c7`; `test/receipt.test.ts=97a13c8545147686369cf7cb5290d69af94408ffc0615984d4a4eef42940d215`.
- Evidence: `test_results/kaggle_cayleypy_results_ingest_npm_gate/outputs_v10/`.

## Concern

The pinned Vitest pool's bundled Miniflare warns that its latest emulated compatibility date is `2025-07-12`, falling back from the Worker pin `2026-07-28`. The gate passed; this remains a local-emulation limitation and no dependency upgrade was made in Task 2.
