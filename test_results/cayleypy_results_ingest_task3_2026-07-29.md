# CayleyPy Results Ingest Task 3 test evidence (2026-07-29)

## Result

Exact private CPU Kaggle v21 is GREEN for the current Task 3 executable payload: all six gate commands exited 0, schema 11/11 passed, real-migration receipt 23/23 passed, Worker 41/41 passed, the Worker pool was 64/64, and TypeScript typecheck passed. The downloaded manifest contained 18 inputs and all 18 SHA-256 values matched the current worktree with zero mismatches.

## TDD trail

- v17 preserved behavior-first RED: schema 11/11 and receipt 23/23 passed; Worker test collection and typecheck failed because `src/worker.ts` did not exist.
- v18 and v19 passed typecheck plus the existing suites but exposed the pinned Worker pool's Ajv deep-import collection failure. A generic dependency include in v19 did not change the failure.
- v20 fixed collection with exact optimizer entries and reached 60/64. Its four failures were all test-harness defects; production `worker.ts` and `storage.ts` already had their final v21 hashes.
- v21 corrected those tests and passed every gate.

## Verified contract

- Exact fail-closed mode resolver; reject/missing/empty/mixed/unknown touch no body, rate binding, R2, D1, or Queue.
- 25 MiB declared and streamed body limits, strict JSON/schema handling, and no more than 100 envelopes.
- Safe normal/store-only/duplicate/mixed receipts; store-only has an explicit Queue-free persistence contract.
- Safe status/health surfaces and zero payload/raw-mode/infrastructure logging.
- Optional Cloudflare rate binding plus load-bearing atomic D1 per-IP/global counters, including exactly 30 concurrent per-IP admissions and one concurrent global final-envelope admission.
- Normal-only scheduled recovery with 60-second staleness, a 50-row bound, same-id resends, queued refresh/error clearing, store-only resumption, and retry failure fairness.

## Exact v21 payload anchors

- `migrations/0002_ingest_rate_limits.sql`: `803e8b3af0dc4e1af2ddd32301a14fdba9d543cf9d5512a934f2d581f85e28f4`
- `src/storage.ts`: `929a71468e5f5773e1c6aad8416360a93e46213b5f715819dd8bdd9fc9338c8b`
- `src/worker.ts`: `085838c41e6b1f8286bcbd83c632e869e446746672f407f64822b4d544ba898b`
- `test/worker.test.ts`: `d0f61086053ab8a9287e597392d5290dab00db22e59d48284640421c3b78d89b`
- `vitest.config.ts`: `06efb33ebd896711ccf3c2331417ed24b3fe44a46a485ec9b652a444402fd895`
- `wrangler.jsonc`: `66c6bbde521691a81f28d7861c3e8e1800cefd0d9a7fa6cdd1f0da4e67d918cd`

## Evidence artifact SHA-256

- `outputs_v21/payload-sha256.json`: `6555a361bcd9b46bb73b4d6dfeb5bf07117a0f86254bb9cdcd3d9a2ca4b2a62b`
- `outputs_v21/npm-test.log`: `43f2914fb8db9a3f037fc82357ec9064f02b6ebdb36873b2bd814cc773d52759`
- `outputs_v21/npm-typecheck.log`: `8fa1cf5506304e8abac55868e7f1a136c9b1dde57a3981a382da4c21ea129a6f`
- `outputs_v21/npm-gate-results.json`: `fe1b97f873759e337e468d312e19211871455286ad403a3c4a70e1d4103a9274`
- `pulled_v21/kernel-metadata.json`: `572d93ed4dd74d3aeabba1951ab4e098cbf6d217219c5b9edddab17bdbb35ae0`
- `pulled_v21/cayleypy-results-ingest-npm-gate.ipynb`: `a0a4a7a68b50433093f89ee659bd4cf1a19d14245aaf97bfa683b647d3669e5d`

## Metadata and limitations

Pulled v21 metadata confirms `trydotatwo/cayleypy-results-ingest-npm-gate`, `is_private=true`, CPU-only (`enable_gpu=false`, `enable_tpu=false`, `machine_shape=None`), and Internet enabled. The pinned test runtime falls back from compatibility date `2026-07-28` to supported `2025-07-12`. Wrangler remains pinned to 4.26.0, below the documented 4.36.0 minimum for Rate Limiting binding configuration, so no such resource/binding was configured or created; the tested D1 limiter is authoritative. The v21 manifest hashes exact pre-filter worktree bytes; Git normalizes CRLF to LF for 11 staged text inputs, and a normalized-content audit found zero non-EOL differences.

No Task 4 consumer/replay/writer, deployment, resource or secret creation, git push, GitHub mutation, or public publication was performed.
