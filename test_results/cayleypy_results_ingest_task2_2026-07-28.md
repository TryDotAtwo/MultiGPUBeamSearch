# CayleyPy Results Ingest Task 2 (2026-07-28)

Private CPU Kaggle gate v8 passed with Node v20.19.0/npm 10.8.2: lock generation, npm ci, 11 Node schema tests, 5 Miniflare D1/R2 receipt tests, and TypeScript typecheck. It verifies R2 raw write before D1 receipt, D1 `received` before awaited Queue durability, and receipt only after `queued`; Queue failure leaves raw object plus retryable row. Evidence: `test_results/kaggle_cayleypy_results_ingest_npm_gate/outputs_v8/`.

Miniflare reports a fallback from the configured 2026-07-28 compatibility date to 2025-07-12; this is recorded, not suppressed.
