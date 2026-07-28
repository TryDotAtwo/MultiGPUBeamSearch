# CayleyPy Results Ingest Task 2 review fix (2026-07-29)

Private CPU Kaggle gate v10 passed with Node v20.19.0/npm 10.8.2: lock generation, npm ci, 11 Node schema tests, 15 real Miniflare D1/R2 receipt tests, and TypeScript typecheck.

The v10 tests prove bounded duplicate waiting through the final reread, one durable Queue write, cleanup of only the D1 conflict loser's raw object, raw retention after an ambiguous D1 insert failure, safe cleanup failure, checked transition-false handling, and separation of Queue from D1 errors. Exact evidence is under `test_results/kaggle_cayleypy_results_ingest_npm_gate/outputs_v10/`; `src/storage.ts` SHA-256 is `ecd55cb4088a35b9672314e149d44bff83c5af69316dff3d87d78581d1e0a2c7` and `test/receipt.test.ts` SHA-256 is `97a13c8545147686369cf7cb5290d69af94408ffc0615984d4a4eef42940d215`.

Miniflare reports a fallback from configured compatibility date 2026-07-28 to 2025-07-12; this is recorded and not suppressed.
