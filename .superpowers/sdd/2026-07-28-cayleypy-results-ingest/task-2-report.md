# CayleyPy Results Ingest Task 2 round-2 verification (2026-07-29)

## Delivered

- Closed the D1/Queue crash window without claiming exactly-once delivery. A duplicate that remains `received` after its final bounded reread resends the same `{submission_id}` and then confirms `queued`/later or `retryable`; a normal concurrent duplicate still observes one Queue send when the winner completes inside the wait.
- Added `recoverStaleSubmissions(env, { staleBefore, limit })`. It queries a bounded `received|retryable` page through `submissions(state,updated_at)`, resends the existing submission id, retains raw R2, and uses checked state transitions.
- Queue-send success compare-transitions `received|retryable -> queued` and clears stale `safe_error`. Queue-send failure first accepts consumer progress already at queued-or-later; otherwise it compare-transitions `received|retryable -> retryable`, increments `retry_count`, refreshes `updated_at`, and records only `queue_unavailable`.
- Failed retry metadata gives the bounded recovery query fairness: a failed first page moves behind untouched stale rows rather than hot-looping forever.
- Updated the architecture and implementation plan contract. Task 3 must expose `scheduled(controller, env, ctx)` with `RECOVERY_STALE_MS = 60_000` and `RECOVERY_LIMIT = 50`; Task 4 must preserve that handler and idempotently compare-transition `received|queued|retryable -> validating` because Queue delivery may beat the producer's post-send `queued` update.
- Real Miniflare coverage now includes immediate-consumer interleaving, ambiguous Queue failure after consumer progress, crash-before-send scheduled recovery, stale duplicate resend, retryable sweeping, one-row recovery duplication, retry-page fairness beyond `limit`, stale-error clearing, normal one-enqueue duplicates, raw retention, and prior concurrency/cleanup gates.

## Private CPU runtime gate

- Kernel: `trydotatwo/cayleypy-results-ingest-npm-gate`, version 12, private, CPU-only, Internet enabled.
- Pulled metadata confirms `is_private=true`, `enable_gpu=false`, and the expected kernel id. The pulled v12 notebook carries the same 14-file `EXPECTED_SHA256` map as the locally generated source.
- Terminal status: `KernelWorkerStatus.COMPLETE`.
- Node `v20.19.0`, npm `10.8.2`.
- `npm install --package-lock-only --no-audit --no-fund`: PASS.
- `npm ci --no-audit --no-fund`: PASS.
- `npm test`: PASS - 11 schema tests plus 18 real Miniflare D1/R2 receipt/recovery tests.
- `npm run typecheck`: PASS.
- Downloaded `payload-sha256.json` matches all 14 current worktree inputs byte-for-byte. Key hashes: migration `8c3fe6fdc4381e123a901593962194f90aae7bfc484e6a6f5685195d05cb0ba6`; `src/db.ts` `a16b27d6c09808a22a52824f9a1b8997b86e6898c8b6d04463df8c4551378221`; `src/storage.ts` `5727b582b69318766f07c60f859a7603097922572df9bfa8723ffb5e3f8e6553`; `test/receipt.test.ts` `c6215277858f55d8422e23244254a4761b3ba802efde4b99f7fc8ac645120698`.
- The downloaded regenerated lockfile is text-equivalent to the embedded lockfile; its byte hash differs only because npm normalized CRLF to LF (`6c8415...` input, `fec430...` output). The exact input hash is present in the 14/14 payload proof, and `npm ci` used the regenerated lock successfully.
- Evidence: `test_results/kaggle_cayleypy_results_ingest_npm_gate/outputs_v12/` and `test_results/kaggle_cayleypy_results_ingest_npm_gate/pulled_v12/`.

## v11 diagnostic run

Version 11 remained `KernelWorkerStatus.RUNNING` across repeated polls through at least `2026-07-29 00:55:20 +03:00`. A live `kaggle kernels output` attempt at that time returned no files. The identified cause was test-only microtask starvation: a new concurrency test polled with `await Promise.resolve()` and could prevent timers/runtime progress indefinitely. No production code used that loop. The test was replaced with explicit deferred Queue-start signals, and v11 was superseded by exact private v12. Kaggle CLI exposes only the latest version's logs after supersession, so no v11 terminal artifact was available.

## Runtime limitation

The pinned Vitest pool's bundled Miniflare warns that its latest emulated compatibility date is `2025-07-12`, falling back from the Worker pin `2026-07-28`. The v12 gate passed; this remains a local-emulation limitation and no dependency upgrade was made in Task 2.
