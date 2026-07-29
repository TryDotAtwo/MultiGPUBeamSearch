# CayleyPy Results Ingest Task 3 test evidence (2026-07-29)

## Result

Exact private CPU Kaggle v29 is GREEN for the restored production compatibility date `2026-07-28`: terminal `KernelWorkerStatus.COMPLETE`; all 14 commands exited 0; schema 12/12 passed; the real-migration receipt plus Worker pool passed 70/70 in both the normal npm test and an independent Worker rerun; TypeScript typecheck passed; the exact registry/resolved-stack checks passed; and the compatibility-warning scan found no fallback. The pulled notebook, pre/post-install manifests, and current worktree agree on all 18 payload files.

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
- Wrangler and the Worker test runtime both use the required production compatibility date `2026-07-28`; the exact v29 logs contain no fallback or unsupported-date warning.

## Compatibility-date restoration follow-up

- The prior v25 review result remains valid historical behavior evidence, but its deliberate `2025-07-12` compatibility-date downgrade is superseded. Production and tests are restored to `2026-07-28`; no production Worker, schema, migration, storage, CUDA, or beam-search semantics changed.
- Exact direct pins are `@cloudflare/vitest-pool-workers=0.19.0`, `@cloudflare/workers-types=5.20260729.1`, `vitest=4.1.10`, and `wrangler=4.115.0`. The resolved transitive runtime is `miniflare=4.20260722.1`; a root override pins the only `workerd` path to `1.20260729.1`, because the packages otherwise bundle `workerd=1.20260722.1`, which predates the required date.
- The test configuration uses the current `cloudflareTest(...)` plugin API. Test bindings are typed through `Cloudflare.Env`, and `env` comes from `cloudflare:workers`; these are test-harness compatibility changes only.
- The private gate deterministically bootstraps Node `v22.23.1` from the official archive after verifying SHA-256 `9749e988f437343b7fa832c69ded82a312e41a03116d766797ac14f6f9eee578` against the official `SHASUMS256.txt`. It parses registry JSON from stdout only, scans combined stdout/stderr for compatibility fallbacks, verifies the one-path workerd resolution, and compares all 18 payload hashes again after `npm install --package-lock-only` and `npm ci`.
- Official anchors: the [`@cloudflare/vitest-pool-workers` 0.19.0 release](https://github.com/cloudflare/workers-sdk/releases/tag/%40cloudflare%2Fvitest-pool-workers%400.19.0), its [tagged package metadata](https://github.com/cloudflare/workers-sdk/blob/%40cloudflare%2Fvitest-pool-workers%400.19.0/packages/vitest-pool-workers/package.json), the [`workerd` 1.20260729.1 release](https://github.com/cloudflare/workerd/releases/tag/v1.20260729.1), and Cloudflare's [Vitest integration configuration](https://developers.cloudflare.com/workers/testing/vitest-integration/configuration/).

### Follow-up RED/GREEN trail

- v26 diagnostic RED proved the initial `0.18.8` candidate obsolete: schema remained 12/12, but the Worker configuration failed on the removed `@cloudflare/vitest-pool-workers/config` export; typecheck failed on unavailable old test module types; the first registry parser was contaminated by stderr notice text; and the generated lock differed from the submitted lock. Its resolved override and compatibility-warning checks were already clean.
- v28 ran the exact final package/runtime stack, preserved the 18-file payload through installation, passed schema 12/12 and Worker 70/70 twice, and found no fallback warning. It remained RED only because augmenting the obsolete `ProvidedEnv` shape did not type the new `cloudflare:workers.env` bindings; `tsc --noEmit` exited 2.
- v29 changed only test binding typing/migration-helper compatibility from v28. All 14 commands exited 0, including five exact registry queries, lock-only install, `npm ci`, critical-stack resolution, schema+Worker tests, the independent Worker rerun, typecheck, and the fail-closed gate assertions.

### Exact v29 payload anchors

- `package-lock.json`: `bf087c2dad9320ebb0091990eacded9dd751e4d3b3eb8f1c1ad8c99b12f571e3`
- `package.json`: `80e59683dc86dbeb7d12fb1200da52a1bfd40573c084acee55b41e61fb816929`
- `wrangler.jsonc`: `21018636ac49a9836762f03aa28160e53a9b50cbf4e9e850a5579ca7845a770c`
- `vitest.config.ts`: `81f596085ac0467e05bf8b99976aa1ff17475271151d4725d9a0b7c183084e21`
- `tsconfig.json`: `68f0b9d9dcfcb08ab1d2164b5463cbd670ab6f8ad62f8623a7bb7434d35ea8fa`
- `test/apply-migrations.ts`: `ff6343c058fe61a7b523f98163e9dddaae55941fde5ad416f5212e2da54b3ade`
- `test/receipt.test.ts`: `6841682dbb4d7e03760e6331eca5c614c2c92f9047239bcdb49cf89bfe0efc72`
- `test/worker.test.ts`: `99f0b764682379e2ff16aabc606ce35722f7cea39b716203e0283271d126a4c1`
- Production `src/worker.ts`, `src/schema.ts`, and `src/storage.ts` remain byte-identical at `f45a1fb11fb80c93a872e20645faa3c7e822075b640347b64f3be8a93259dc11`, `5b2d0cdc1736bb5af720624d6f069567975d06991821e6a22776eacc3199126f`, and `929a71468e5f5773e1c6aad8416360a93e46213b5f715819dd8bdd9fc9338c8b`.

## Historical v25 payload anchors

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
- Follow-up diagnostic v26 capture manifest: `control_v26_probe/capture-manifest.json`, SHA-256 `11c2d99951f4573997445d7a94da5ffa82f6b375d5a3d7eab8f1f805b037f3e1`.
- Follow-up type-only RED v28 capture manifest: `control_v28_final/capture-manifest.json`, SHA-256 `bb133bd5ddf2492f002b569544d36e4a3afc9c7c3df90df6bc4474ddade8c03d`.
- Final follow-up GREEN v29 capture manifest: `control_v29_final/capture-manifest.json`, SHA-256 `93916f0701314e977a5d0e352615d396532a547fbabc21faf251312fb37dddce`.
- v26's source pull was rerun after the next push and is explicitly marked stale, excluded from v26 run identity, and validated to differ on exactly `package.json`, `tsconfig.json`, and `vitest.config.ts`. Its downloaded v26 output manifests/logs are the RED identity. v28/v29 pulled notebooks are run-exact; v29's pulled notebook SHA-256 is `e5d0eb88692c1f2ad8e99d959854527685c1881c7e1182889946300df24f2382`.
- The final Git index is byte-exact to v29 for 14 of 18 payload files. The four unchanged tracked inputs `migrations/0001_initial.sql`, `src/db.ts`, `test/schema.test.ts`, and `vitest.schema.config.ts` differ only by Git's configured CRLF-to-LF text normalization; a normalized comparison found zero non-EOL differences.
- Each manifest records kernel ref/version, push/status/pull UTC timestamps, exact retained-file hashes, and semantic validation. Retained raw evidence includes push receipt, COMPLETE status, output/pull receipt, pulled private metadata and notebook, node/npm versions, gate results, payload manifest, schema/Worker logs, and typecheck log.
- v25 pulled notebook SHA-256 is `65e9f119e50f56046c60bc6fce36fe6ecb1c618181f7da51b0d1f51be7db9d92`; the pushed notebook SHA-256 was `3741c70e333a22f93173afb9d7e219a915ca4b71d2f85cbf8a6f0f865c7b3abd`. Kaggle normalizes the notebook wrapper, so payload acceptance is based on parsed embedded-archive semantics, not wrapper byte equality.
- The evidence set is intentionally list-free. The follow-up set excludes v27 intermediate bulk, identity-bearing kernel-list output, aggregate logs, downloaded duplicate lockfiles, and other non-minimal artifacts. The capture validator reports zero secrets, tokens, real local names, or user-home paths across all 69 retained v26/v28/v29 files.

## Metadata and boundary

Pulled v23, v25, v28, and v29 metadata confirm `trydotatwo/cayleypy-results-ingest-npm-gate`, `is_private=true`, CPU-only (`enable_gpu=false`, `enable_tpu=false`, `machine_shape=None`), and Internet enabled. v26's pulled metadata confirms the same execution shape, while its later source pull is not used as v26 payload evidence. No `kaggle kernels list` command was run for the follow-up; raw receipts preserve only the benign local Kaggle CLI outdated-version warning.

No Task 4 consumer/replay/writer, CUDA or beam-search change, deployment, Cloudflare resource/secret creation, git push, GitHub mutation, or public publication was performed.
