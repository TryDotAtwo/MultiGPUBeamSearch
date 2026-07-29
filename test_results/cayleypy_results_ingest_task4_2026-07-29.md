# CayleyPy Results Ingest Task 4 — 2026-07-29

## Outcome

Task 4 is green on the exact private Kaggle CPU gate:

- kernel: `trydotatwo/cayleypy-results-ingest-task-4-exact-gate`
- version: 4
- URL: `https://www.kaggle.com/code/trydotatwo/cayleypy-results-ingest-task-4-exact-gate`
- status: `COMPLETE`
- Node: `v22.23.1`
- npm: `10.9.8`
- schema tests: `10/10`
- Worker-pool tests: `99/99`, then `99/99` again
- TypeScript: `tsc --noEmit` passed
- all 14 gate commands passed
- exact registry versions, resolved dependency stack, workerd override, lockfile, and pre/post-install payload hashes passed
- compatibility-warning scan found no fallback or unsupported-date warnings

The Windows Kaggle CLI downloaded every output file and then returned exit code 1 only because its console codec could not encode a Vitest checkmark. The downloaded JSON result records `all_commands_passed=true`; the CLI encoding failure is not a kernel failure.

## Runtime stack

- `@cloudflare/vitest-pool-workers=0.19.0`
- `@cloudflare/workers-types=5.20260729.1`
- `vitest=4.1.10`
- `wrangler=4.115.0`
- resolved `miniflare=4.20260722.1`
- overridden and resolved `workerd=1.20260729.1`
- compatibility date `2026-07-28`

## Verified behavior

- Immutable R2 body size and UTF-8 size are bounded before replay, and the body SHA-256 must exactly match R2 custom metadata.
- Canonical schema, semantic idempotency, proof hashes, model-head dimensions, state labels, every generator, reflection provenance, inverse naming, and the full submitted path are independently checked server-side.
- Replay is bounded to `state_len<=120`, `moves<=256`, `path<=4096`, and a bounded total proof-cell budget.
- `received|queued|retryable` rows use an idempotent `validating` claim. Duplicate live validating deliveries ACK without R2 or writer access; stale validating rows are recoverable.
- `store_only`, `reject`, missing, and unknown modes durably park recoverable rows before ACK and perform no R2 replay or writer call.
- Transient failures retry with bounded backoff. Exhaustion durably records `dead_letter`, explicitly hands off to the configured DLQ, and retains immutable raw R2 data.
- A validated row never downgrades during an ordinary GitHub-writer retry. Repeated delivery makes the same deterministic writer enqueue.

## RED to GREEN provenance

Version 2 passed the exact stack, typecheck, schema `10/10`, consumer `21/21`, receipt `23/23`, Worker `49/49`, and replay `5/6`. Its only failure was a test fixture with state `[10,20,30,40]` for `state_len=4`; Task 4 correctly rejected those class labels. The fixture was changed to `[0,1,2,3]`, with no production change. Version 3 then passed the complete `99/99` Worker-pool gate twice.

The four new source/test files were mechanically normalized to repository LF line endings before staging. Version 4 repeated the complete exact gate on those final bytes and passed unchanged.

## Evidence

Downloaded version-4 artifacts are under:

`test_results/kaggle_cayleypy_results_ingest_task4_gate/outputs_v4/`

Important records:

- `npm-gate-results.json`
- `npm-test.log`
- `npm-test-worker.log`
- `npm-typecheck.log`
- `resolved-stack.json`
- `registry-metadata.json`
- `compatibility-warning-scan.json`
- `payload-sha256.json`
- `post-install-payload-sha256.json`
- `gate-assertions.log`

Gate package SHA-256:

- notebook: `B966CDCBF202B276B169C9BBE67FE5F7D141320CB27737344287C5E34C72B34F`
- metadata: `F9064F9AF2B77C8066798F0A121E8711F3B8907BA8206CCC1C2004D58B5B01EF`

Selected exact payload SHA-256:

- `src/consumer.ts`: `4b6c8ef7c513205568e59262049d5d7b0488b9666e02ebb46f84dc8c8ce7654c`
- `src/replay.ts`: `0acb7614048ee4f09f16a15306bb13b2ef3524abba0c98f9b8bced231f2f1d16`
- `test/consumer.test.ts`: `5ffd80a1044ceb8b9216376a026fde3a38b89563a2f7b034792627234ed6e51d`
- `test/replay.test.ts`: `460bd24713a7bd180cf758e8ce61f05b5a369ae9c1e12ffa5003405a3a32a805`

## Boundary

No Worker deployment, Cloudflare resource or secret creation, GitHub mutation, public publication, CUDA change, or beam-search architecture change occurred. Task 4 is intentionally not deployed alone: the configured consumer requires the Task 5 serialized GitHub writer binding for validated results.
