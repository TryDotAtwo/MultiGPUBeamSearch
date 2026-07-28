# CayleyPy Results Ingest Task 2 verification (2026-07-28)

## Delivered

- Added migration `0001_initial.sql` for the submission state machine and indexes.
- Added deterministic canonical JSON and SHA-256 semantic idempotency. Client transport fields (`submission_id`, supplied idempotency key, and timestamp) are excluded from semantic identity.
- Raw payload key is service generated as `raw/v1/YYYY/MM/DD/<uuidv7>.json`; the client supplies no storage/repository path. R2 uses `If-None-Match: *`, captures SHA-256 custom metadata, and rejects a conflicting existing object without exposing its contents.
- Receipt order is R2 raw object, D1 `received`, awaited durable Queue write, D1 `queued`, then receipt. Queue failures retain raw R2 and change the row to `retryable` with only safe error code `queue_unavailable`.
- State changes use bound prepared statements and `WHERE submission_id = ? AND state IN (...)`, checking `meta.changes === 1`.

## Private CPU runtime gate

- Kernel: `trydotatwo/cayleypy-results-ingest-npm-gate`, private, CPU-only, Internet enabled, version 8.
- Node `v20.19.0`, npm `10.8.2`.
- `npm install --package-lock-only --no-audit --no-fund`: PASS.
- `npm ci --no-audit --no-fund`: PASS.
- `npm test`: PASS — 11 schema tests in Node plus 5 receipt/order/idempotency/retry/transition tests with real Miniflare D1/R2 bindings.
- `npm run typecheck`: PASS.
- Evidence: `test_results/kaggle_cayleypy_results_ingest_npm_gate/outputs_v8/`.

## Concern

The pinned Vitest pool's bundled Miniflare warns that its latest emulated compatibility date is `2025-07-12`, falling back from the Worker pin `2026-07-28`. The gate passed; this is retained as a local-emulation limitation and no dependency upgrade was made in Task 2.
