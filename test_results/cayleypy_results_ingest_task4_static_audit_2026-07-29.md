# CayleyPy Results Ingest Task 4 static audit ? 2026-07-29

Scope: uncommitted Queue consumer/replay diff only. No CUDA or beam-search source was inspected or changed.

## Contracts checked

- State replay uses pull permutations and rejects `state_len < 1` or `> 120`; padding bytes are not represented in the proof.
- The Worker resolves `INGEST_MODE` before iterating Queue messages. In non-normal modes the consumer parses only a UUIDv7 `submission_id`, reads only D1, parks recoverable rows to `retryable/ingest_paused`, then ACKs. It performs no R2 replay or writer call.
- Normal delivery compare-transitions `received|queued|retryable -> validating`; raw R2 is parsed and schema/proof/replay/idempotency checks are repeated server-side.
- Invalid immutable raw becomes `rejected/invalid_envelope` and ACKs. Transient D1/R2 failure becomes retryable with bounded exponential delay.
- Attempts at the configured limit transition to `dead_letter` and call `message.retry()` without a delay so Cloudflare moves the retained message to the configured DLQ.
- `validated` is intentionally not terminal for normal Queue processing. A delivery calls the deterministic GitHub Writer DO. If enqueue or its D1 confirmation is ambiguous, the row remains `validated` (or reaches `dead_letter` only at exhaustion), the safe error is recorded, and the Queue retry will repeat the same idempotent enqueue.
- A successful writer enqueue self-transitions `validated -> validated` to clear stale safe error. Final repository states ACK without touching R2/writer.
- Task 3 scheduled recovery remains scoped to `received|queued|retryable`; Task 4 test coverage explicitly preserves `validating`, `validated`, and `dead_letter`.

## Deterministic test additions

`test/consumer.test.ts` covers valid replay, malformed immutable raw, both non-normal modes, poison/terminal deliveries, state length 121, unknown moves, transient R2 and D1, bounded retry delay, retry exhaustion/DLQ state, idempotent duplicate writer delivery, and scheduled-state preservation.

## Verification status

- `git diff --check`: clean except existing CRLF normalization warnings.
- Local npm is unavailable: Node `v22.22.0` / npm `10.9.4` exits `Exit handler never called!` before creating `node_modules/.bin/tsc`. No local TypeScript or Vitest result is claimed.
- Prepared private CPU Kaggle package: `test_results/kaggle_cayleypy_results_ingest_task4_gate/kernel/`. It contains checksum-pinned exact-stack inputs, no secrets, GPU disabled, Internet enabled, and has only been locally AST/JSON-built; it has not been pushed.

## Superseding runtime verification

The prepared gate was subsequently pushed privately. Version 2 preserved a single test-fixture RED after the stricter permutation-class contract correctly rejected labels outside `0..state_len-1`. The fixture alone was corrected. Version 3 passed; after mechanical LF normalization, version 4 repeated the exact green gate against the bytes staged for commit. See `test_results/cayleypy_results_ingest_task4_2026-07-29.md`.
