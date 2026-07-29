# CayleyPy Results Ingest Task 3 verification (2026-07-29)

## Delivered scope

- Added `POST /v1/results`, `GET /v1/submissions/:id`, and `GET /healthz` in a Worker entry point that exports only `fetch` and `scheduled`; Task 4 Queue consumption is not implemented.
- Enforced the exact case-sensitive `INGEST_MODE=normal|store_only|reject` allowlist. Missing, empty, mixed-case, and unknown values resolve to `reject` without echoing or logging the raw value, and reject before body consumption, rate infrastructure, R2, D1, or Queue access.
- Added strict `application/json` handling, a `Content-Length` precheck, incremental fatal UTF-8 decoding with a 4 MiB hard bound, strict schema validation, and the existing 100-envelope maximum. The streaming path no longer retains byte chunks or allocates a second full byte buffer, clears the decoded string after parsing, and schema validation skips `JSON.stringify` when ingress supplied the exact byte length.
- `normal` persists through the Task 2 R2-before-D1 contract and publishes one Queue message per accepted envelope. `store_only` uses the explicit `receiveEnvelopeStoreOnly` persistence path, whose environment type has no Queue and whose implementation performs zero Queue access while retaining semantic idempotency.
- `POST` returns `202` with safe receipts and safe per-index failures for mixed persistence/Queue outcomes. Receipt bodies expose only `submission_id`, `idempotency_key`, and the service-generated status URL.
- Status exposes only receipt state fields and a safe `404`; health exposes only the resolved mode, exact public limits, and recovery constants. No payload, binding identifier, secret, or raw mode is logged.
- Added append-only migration `0002_ingest_rate_limits.sql`. Each D1 counter consumption samples the current minute and uses one monotonic conditional UPSERT: equal windows increment, newer windows reset, and stale windows make zero changes and are rejected. D1 remains the load-bearing limiter at 30 requests/minute/IP and 2,000 envelopes/minute globally; the optional Cloudflare binding is only an additional early per-IP rejection path.
- Added the one-minute cron trigger. Scheduled recovery is a strict no-op outside `normal`; normal mode directly awaits the existing bounded recovery helper with a 60-second stale cutoff and limit 50.
- Per-envelope persistence is bounded to eight concurrent operations while preserving response order. Normal-mode duplicate polling receives a deterministic request-wide budget of 400 rereads through `RequestMeta`; the 100-existing-duplicate regression proves exactly 702 internal D1/Queue terminal calls, below the requested 1,000-call ceiling.
- Malformed percent encoding on the public status route returns a value-free `400 invalid_submission_id` before D1 access.
## Behavior and concurrency coverage

The Worker suite has 47 tests. It covers exact mode resolution; fail-closed early rejection; methods and media types; malformed JSON, strict-schema, oversized declared, and oversized streamed bodies; the 100-envelope cap; normal/store-only/duplicate/mixed outcomes; status and health data minimization; absence of logs; the real migration; optional binding rejection; D1 per-IP/global limits and minute rollover; all scheduled mode gates; bounded stale/fresh/queued/retryable recovery; same-id store-only resumption; failure fairness; concurrent store-only duplicates; exactly 30 concurrent per-IP admissions; and atomic rejection of a concurrent global final envelope.

The existing 23-test real-migration receipt suite remains in the same Worker pool, for 70/70 Worker-pool tests. The separate schema suite is 12/12.

## TDD and correction trail

### Private CPU v17 RED

- Kernel `trydotatwo/cayleypy-results-ingest-npm-gate`, immutable version 17, completed privately on CPU with Internet enabled and GPU/TPU disabled.
- Lock generation and `npm ci` passed. The 11 schema tests and 23 existing receipt tests passed.
- The new Worker suite and typecheck failed only because `src/worker.ts` did not exist, preserving the behavior-first RED gate.
- RED payload anchors: `test/worker.test.ts=a20cd54676d2f884f8175b292eac6e146d330c04d8b7a43f43e5d85b13ffb175`, `vitest.config.ts=ec211eb6988bba7a27db6c5c6b408740c6ab0c27a2e03b58811b20c55c605d4e`.

### Private CPU v18-v20 runtime corrections

- v18 implemented the Worker and passed typecheck, schema 11/11, and receipt 23/23, but the Worker suite could not collect because the pinned Workers pool interpreted Ajv's deep CommonJS JSON import as JavaScript.
- v19 tested a generic optimizer include for `ajv` and `ajv-formats`; it reproduced the same `ajv/dist/core.js:21` parse failure and therefore disconfirmed that hypothesis.
- v20 used exact deep-entry optimization for `ajv/dist/2020.js` plus `ajv-formats`. Collection succeeded and 60/64 Worker-pool tests passed. The four failures were test-harness errors: three calls intended to pass `INGEST_MODE=undefined` were defaulted to `normal`, and one test counted the platform's eager stream pull instead of asserting the request remained unconsumed. The production `worker.ts` and `storage.ts` hashes in v20 already equal v21; only the tests were corrected.

## Exact private CPU v21 GREEN

- Terminal status: `KernelWorkerStatus.COMPLETE`.
- Metadata: expected kernel id, `is_private=true`, `enable_gpu=false`, `enable_tpu=false`, `machine_shape=None`, `enable_internet=true`.
- Node/npm version checks, lock-only install, `npm ci`, `npm test`, and `npm run typecheck` all exited 0; `all_commands_passed=true`.
- Test result: schema 11/11; real-migration receipt 23/23; Worker 41/41; Worker pool 64/64; typecheck PASS.
- All 18 downloaded payload hashes were compared to the exact current worktree inputs: 18 checked, zero mismatches.

Key v21 payload SHA-256 values:

- `migrations/0002_ingest_rate_limits.sql`: `803e8b3af0dc4e1af2ddd32301a14fdba9d543cf9d5512a934f2d581f85e28f4`
- `src/storage.ts`: `929a71468e5f5773e1c6aad8416360a93e46213b5f715819dd8bdd9fc9338c8b`
- `src/worker.ts`: `085838c41e6b1f8286bcbd83c632e869e446746672f407f64822b4d544ba898b`
- `test/worker.test.ts`: `d0f61086053ab8a9287e597392d5290dab00db22e59d48284640421c3b78d89b`
- `vitest.config.ts`: `06efb33ebd896711ccf3c2331417ed24b3fe44a46a485ec9b652a444402fd895`
- `wrangler.jsonc`: `66c6bbde521691a81f28d7861c3e8e1800cefd0d9a7fa6cdd1f0da4e67d918cd`

Preserved v21 evidence SHA-256:

- `outputs_v21/payload-sha256.json`: `6555a361bcd9b46bb73b4d6dfeb5bf07117a0f86254bb9cdcd3d9a2ca4b2a62b`
- `outputs_v21/npm-test.log`: `43f2914fb8db9a3f037fc82357ec9064f02b6ebdb36873b2bd814cc773d52759`
- `outputs_v21/npm-typecheck.log`: `8fa1cf5506304e8abac55868e7f1a136c9b1dde57a3981a382da4c21ea129a6f`
- `outputs_v21/npm-gate-results.json`: `fe1b97f873759e337e468d312e19211871455286ad403a3c4a70e1d4103a9274`
- `pulled_v21/kernel-metadata.json`: `572d93ed4dd74d3aeabba1951ab4e098cbf6d217219c5b9edddab17bdbb35ae0`
- `pulled_v21/cayleypy-results-ingest-npm-gate.ipynb`: `a0a4a7a68b50433093f89ee659bd4cf1a19d14245aaf97bfa683b647d3669e5d`

Preserved v17 evidence SHA-256:

- `outputs_v17/payload-sha256.json`: `ad6d0e85055351350783ac00c95d2dba778acceee8d56610aba3c342c222cd1a`
- `outputs_v17/npm-test.log`: `1b840b3a4afb4043c51e2c706c5bb1a1e8b4518632e3697219ca92679251686f`
- `outputs_v17/npm-typecheck.log`: `dd0f7e0c05e6a5f1850c190e1a4ce41b255db31df9cd63e64dd121ec942e9556`
- `outputs_v17/npm-gate-results.json`: `81d8e68f2f6f154719e94b7d5e62a55e40d2021abc07533a7417baca94e4f727`
- `pulled_v17/kernel-metadata.json`: `572d93ed4dd74d3aeabba1951ab4e098cbf6d217219c5b9edddab17bdbb35ae0`
- `pulled_v17/cayleypy-results-ingest-npm-gate.ipynb`: `da72dee2264cb24c9d0a2864a8b56f19a9bc1e28fc9cdb3eff7f85ec760363e2`

## Review hardening: private CPU v22-v25

### Review RED trail

- v22 was the first review-test payload. It preserved three exact schema failures, but an accidental test-helper reference caused a TypeScript harness error before the Worker suite could be accepted. It is retained locally only as diagnostic evidence (`outputs_v22/npm-gate-results.json` SHA-256 `63a57db678988ffdfa1f2ea9400ec962272dbd864ae80fe7e4711029a27a28c9`); it is not part of the committed raw evidence set.
- Corrected private v23 is the accepted review RED. Its push receipt records version 23, status is `KernelWorkerStatus.COMPLETE`, pulled metadata is private CPU-only, and the embedded 18-file archive, notebook `EXPECTED_SHA256`, downloaded payload manifest, and gate-result manifest agree exactly. Schema was 3 failed / 9 passed; the Worker pool was 6 failed / 64 passed; typecheck exited 0. The six Worker failures were exactly the 4 MiB stream bound, malformed percent handling, slow-boundary clock sample, stale counter rollback, unbounded envelope concurrency, and duplicate reread budget.
- v23 raw evidence is committed under `control_v23_red` and `outputs_v23_red`. Its capture manifest hashes every retained file and records push/status/pull timestamps. The pulled notebook SHA-256 (after Kaggle source normalization) is `aaf6e671219a46a4c4b77e95a31acec4e9b812efdec2d5045bfd178f49440f0b`; semantic extraction proves its embedded archive matches the pushed payload. Other key hashes: schema log `a7fcfa5b260acda08ba599cdde6d4001fcdbd2ffa7b8532fa96dfc33a1eb8491`, Worker log `9b4203d54d519755a4bcf938eec49ae7621a07a483f83df9eca7eb1cf9676530`, and payload manifest `7c3996f2d683b2fc56b09b9c7b85b810fe31437bdfa514358f190b1ffee6d055`.

### Implementation and final GREEN

- The rate limiter now samples time inside each D1 counter consumption and refuses stale-window writes. Adversarial tests cover a binding-delayed pre-boundary request interleaved with a current request and a bound stale global UPSERT released after a newer write.
- Request parsing incrementally decodes fatal UTF-8, enforces 4 MiB while streaming, avoids the retained-chunks plus second-buffer peak, clears avoidable text references, and avoids schema reserialization when raw length is supplied. Exact-limit split-multibyte and one-byte-over-limit streams are covered.
- Envelope work uses an order-preserving concurrency bound of eight. A request-wide 400-reread budget is divided across the batch and passed through `RequestMeta`; the 100-received-duplicate gate asserts exactly 702 terminal binding calls. Malformed status path escapes fail with a safe 400 before D1.
- v24 proved the production fix was already complete: schema was 12/12 and 68/70 Worker-pool tests passed. Its two failures were test oracles only (client transport key instead of canonical semantic idempotency and the default 5-second emulator timeout); v24 is summarized locally by `outputs_v24_green/npm-gate-results.json` SHA-256 `4c624328b38ee34287413034b37b73867ae56b40a1bc9c88a981b4f3589efa3a` and Worker log SHA-256 `272d39e5fa7f17b94966301927ea6e4afbd282b44ddc69709fe6c1b0126315cc`.
- Exact private v25 is GREEN and terminal `COMPLETE`. All seven gate commands exited 0; schema is 12/12; the combined receipt+Worker pool is 70/70 in both `npm test` and the independent Worker rerun; typecheck passed. The final duplicate test asserts 702 calls and uses a 30-second test-harness timeout because the remote Miniflare D1 emulator takes about 30 seconds for the bounded 100-item case.
- A fresh semantic audit parsed the pulled v25 notebook, decoded its embedded archive in memory, and compared all 18 archive entries against notebook expectations, downloaded manifests, and the current worktree: zero mismatches. Pulled metadata confirms the expected kernel id, `is_private=true`, GPU/TPU disabled, `machine_shape=None`, and Internet enabled. The committed evidence is deliberately list-free and contains no identity-bearing `list.csv`.
- v25 raw evidence is committed under `control_v25_green` and `outputs_v25_green`. The pushed notebook SHA-256 was `3741c70e333a22f93173afb9d7e219a915ca4b71d2f85cbf8a6f0f865c7b3abd`; the pulled notebook SHA-256 after Kaggle source normalization is `65e9f119e50f56046c60bc6fce36fe6ecb1c618181f7da51b0d1f51be7db9d92`. Semantic extraction, rather than byte equality between those notebook wrappers, proves the embedded 18-file payload is exact. Other key hashes: gate results `8ce3402e5f1d9b42d9e3a556ec5b6d560912e020f669a34cc59ccc795c880b5d`, payload manifest `8a57b47803c610e3ba8bda75ac639393b52c078f07291c60c6200307e8917b44`, combined test log `5e0c4d27650a8d411fe9cf0efa9ad376bf5d6d351741c10abe1769a455426300`, explicit Worker log `e0e57a2e67a5310b4306fddeb64cb4695d6490740fc01fb9c4f6737c4466d748`, and typecheck log `8fa1cf5506304e8abac55868e7f1a136c9b1dde57a3981a382da4c21ea129a6f`.
## Source basis and explicit limitations

- Cloudflare documents request bodies as `ReadableStream` values and its Streams API as the bounded streaming mechanism used here: https://developers.cloudflare.com/workers/runtime-apis/request/ and https://developers.cloudflare.com/workers/runtime-apis/streams/.
- Cloudflare's Rate Limiting binding is location-local rather than a global counter, and its documented Wrangler configuration requires Wrangler 4.36.0 or later: https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/. This project is intentionally pinned to Wrangler 4.26.0, so Task 3 defines the optional runtime binding interface but does not configure, provision, or invent a Rate Limiting resource. The tested atomic D1 counters remain load-bearing.
- The scheduled entry point follows Cloudflare's `scheduled()` handler and `controller.scheduledTime` contract: https://developers.cloudflare.com/workers/runtime-apis/handlers/scheduled/.
- The exact dependency optimization follows the pinned Workers Vitest module-resolution guidance and the documented Vitest/Vite optimizer deep-import controls: https://developers.cloudflare.com/workers/testing/vitest-integration/known-issues/, https://v3.vitest.dev/config/, and https://main.vite.dev/config/dep-optimization-options.
- The deployed Wrangler configuration and the pinned Worker-pool emulator now both use the exact supported compatibility date `2025-07-12`; v25 contains no compatibility-date fallback warning. No untested 2026 runtime claim or dependency upgrade was introduced.
- Kaggle's Windows CLI downloaded the requested artifacts before returning exit 1 while printing a Unicode checkmark. The downloaded JSON and files were complete and inspected directly.
- The v21 SHA-256 manifest records exact pre-filter worktree bytes. Git's configured text filter normalizes CRLF to LF for 11 of the 18 staged text inputs; a staged-versus-tested audit found zero non-EOL content differences.

## Boundary audit

No Task 4 consumer/replay, GitHub writer, queue handler, deployment, Cloudflare resource, secret, git push, GitHub mutation, or public publication is part of this commit. The cron is configuration only; no infrastructure was created. Private Kaggle v17-v25 supplied dependency/runtime evidence only; accepted review RED is v23 and final GREEN is v25.
