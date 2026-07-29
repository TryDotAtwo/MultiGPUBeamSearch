# CayleyPy Results Ingest Task 3 test evidence (2026-07-29)

## Result

Exact private CPU Kaggle v25 is GREEN for the hardened Task 3 payload: terminal `KernelWorkerStatus.COMPLETE`; all seven gate commands exited 0; schema 12/12 passed; the real-migration receipt plus Worker pool passed 70/70 in both the normal npm test and an independent Worker rerun; and TypeScript typecheck passed. The pulled notebook's embedded 18-file archive, its expected hashes, both downloaded manifests, and the current worktree agree with zero semantic payload mismatches.

## TDD trail

- Original Task 3: v17 preserved the missing-Worker RED; v18-v20 isolated Worker-pool Ajv resolution and test-harness defects; v21 passed schema 11/11, receipt 23/23, Worker 41/41, Worker pool 64/64, and typecheck.
- Review v22 preserved the three intended schema RED failures but was superseded because a test helper had one unintended TypeScript reference. Its diagnostic gate-results SHA-256 is `63a57db678988ffdfa1f2ea9400ec962272dbd864ae80fe7e4711029a27a28c9`.
- Corrected review v23 is the accepted RED: schema 3 failed / 9 passed, Worker pool 6 failed / 64 passed, and typecheck exit 0. The failures exactly covered the 4 MiB body limit, malformed percent encoding, two adversarial minute-window races, envelope concurrency, and the duplicate subrequest budget.
- v24 had the final production hashes and reached schema 12/12 plus Worker pool 68/70. Its two remaining failures were test oracles: response order was compared to client transport keys rather than canonical semantic hashes, and the bounded 100-item Miniflare case exceeded Vitest's default five-second timeout. Gate-results SHA-256 is `4c624328b38ee34287413034b37b73867ae56b40a1bc9c88a981b4f3589efa3a`.
- v25 corrected only those test oracles, asserted the exact 702 binding calls, and passed every gate.

## Verified current contract

- The request body has an exact 4 MiB declared/streamed limit. It is decoded incrementally with fatal UTF-8 handling, without retaining byte chunks or allocating a second full byte buffer; the parsed path skips duplicate schema serialization when raw byte length is known.
- D1 rate windows are sampled at each counter consumption. Equal windows increment, newer windows reset, and stale writes change zero rows and return `429`; slow-boundary and stale-run interleavings are tested.
- Per-envelope storage concurrency is at most eight and response order is stable. A request-wide budget of 400 duplicate rereads is passed through `RequestMeta`; 100 existing `received` duplicates use exactly 702 D1/Queue terminal calls, below 1,000.
- Malformed percent escapes return `400 invalid_submission_id` before D1. The existing fail-closed mode, safe receipt/status/health, 100-envelope, authoritative per-IP/global rate, and normal-only recovery contracts remain covered.
- Wrangler and the pinned Worker pool both use the exact supported compatibility date `2025-07-12`; the final logs contain no fallback warning.

## Exact v25 payload anchors

- `src/schema.ts`: `5b2d0cdc1736bb5af720624d6f069567975d06991821e6a22776eacc3199126f`
- `src/storage.ts` (unchanged): `929a71468e5f5773e1c6aad8416360a93e46213b5f715819dd8bdd9fc9338c8b`
- `src/worker.ts`: `f45a1fb11fb80c93a872e20645faa3c7e822075b640347b64f3be8a93259dc11`
- `test/schema.test.ts`: `28ff4675db6dc1be505e9d9591f0491e84dab8d4cf6b7fad24895fab79f4900c`
- `test/worker.test.ts`: `4a3f514a3e33f47599aa529c72e175d1211d3b74962518c0105dee1c89363ddc`
- `vitest.config.ts`: `a63ad781e2a6832f52ab0d0ba0e6a2950595492fd12883fa4ecb71f212d790e6`
- `wrangler.jsonc`: `fbca840524e5c93b7805418e5046f83bbc89fde2ecca272c99be2011c93625d7`

## Committed raw evidence

- Accepted v23 RED capture manifest: `control_v23_red/capture-manifest.json`, SHA-256 `a11ab20590ec7701a5ef89b90e3b62fa04216f049248e15e622a07c98e95daca`.
- Final v25 GREEN capture manifest: `control_v25_green/capture-manifest.json`, SHA-256 `e8c8b612c1df1707ea5018167a9463472cf496215fe8b01eb888ef3325dfc11d`.
- Each manifest records kernel ref/version, push/status/pull UTC timestamps, exact retained-file hashes, and semantic validation. Retained raw evidence includes push receipt, COMPLETE status, output/pull receipt, pulled private metadata and notebook, node/npm versions, gate results, payload manifest, schema/Worker logs, and typecheck log.
- v25 pulled notebook SHA-256 is `65e9f119e50f56046c60bc6fce36fe6ecb1c618181f7da51b0d1f51be7db9d92`; the pushed notebook SHA-256 was `3741c70e333a22f93173afb9d7e219a915ca4b71d2f85cbf8a6f0f865c7b3abd`. Kaggle normalizes the notebook wrapper, so payload acceptance is based on parsed embedded-archive semantics, not wrapper byte equality.
- The evidence set is intentionally list-free. It excludes identity-bearing kernel-list output, zero-byte main logs, downloaded duplicate lockfiles, and invalid/partial bulk from v22/v24. A staged-evidence scan is required to show no secrets, tokens, real local names, or user-home paths.

## Metadata and boundary

Pulled v23 and v25 metadata confirm `trydotatwo/cayleypy-results-ingest-npm-gate`, `is_private=true`, CPU-only (`enable_gpu=false`, `enable_tpu=false`, `machine_shape=None`), and Internet enabled. The Windows Kaggle CLI downloaded every output before its console failed to encode a Unicode checkmark; the output-download receipts preserve that bounded known issue and the files were inspected directly.

No Task 4 consumer/replay/writer, CUDA or beam-search change, deployment, Cloudflare resource/secret creation, git push, GitHub mutation, or public publication was performed.