# CayleyPy results ingest Task 5 verification (2026-07-29)

## Outcome

Task 5 implements the final serialized GitHub publication boundary without changing the beam-search or CUDA architecture.

The Worker now uses a repository-scoped GitHub App installation token and a single SQLite-backed `GitHubWriter` Durable Object. Validated submission ids are stored durably before publication. The writer then re-reads and verifies D1 provenance plus immutable R2 bytes, generates a server-owned append-only path, and commits to `ingest/staging` with non-force reference updates.

## Reliability contract

- Exact modes remain `normal`, `store_only`, and `reject`; missing or unknown values fail closed.
- The writer rechecks `normal` immediately before token acquisition and before every GitHub API request.
- One Durable Object serializes Git history updates from concurrent Queue deliveries.
- Each flush is bounded to 100 records and 5 MiB.
- Identical existing content converges without another commit.
- Different existing content is terminalized as `publication_path_conflict`; it is never overwritten.
- Git reference conflicts retry up to three times with `force: false`.
- Ambiguous or transient failures retain pending work and force a future alarm even when the current alarm is executing.
- Missing/non-validated poison ids are removed from the bounded pending window, so 100 early stale keys cannot starve later validated results.
- R2 digest/path conflicts retain raw bytes for forensic recovery and move the D1 row to `dead_letter`.
- The GitHub client discards response bodies on failure and emits only stable safe error codes; tokens and private-key bytes are not logged.

## Declarative Durable Object configuration

`wrangler.jsonc` uses the current declarative lifecycle:

```json
"exports": {
  "GitHubWriter": { "type": "durable-object", "storage": "sqlite" }
}
```

The legacy `migrations` array is intentionally absent because current Wrangler makes `exports` and `migrations` mutually exclusive. Named staging and production environments repeat all non-inheritable bindings and use isolated D1, R2, and Queue resource names. Staging starts in `store_only`; production starts in `reject`.

## Exact private Kaggle gate

Kernel: `trydotatwo/cayleypy-results-ingest-task-5-exact-gate`

Final version: **v9**, private CPU, Internet enabled.

Runtime:

- Node `v22.23.1`
- npm `10.9.8`
- `@cloudflare/vitest-pool-workers=0.19.0`
- `@cloudflare/workers-types=5.20260729.1`
- `vitest=4.1.10`
- `wrangler=4.115.0`
- resolved `miniflare=4.20260722.1`
- sole `workerd=1.20260729.1`

Final evidence:

- all 14 gate commands passed;
- TypeScript `tsc --noEmit` passed;
- schema plus Wrangler config: 16/16 passed;
- full real Worker/Miniflare pool: 119/119 passed twice;
- 100 concurrent duplicate Durable Object RPCs converged to 100 durable pending records;
- 100 lexicographically early poison ids did not starve a later validated record, which reached `staged`;
- a transient alarm-time GitHub failure left a future alarm scheduled;
- immutable R2 tampering terminalized without deleting raw evidence;
- pre-install and post-install payload hashes matched exactly;
- compatibility-warning scan was empty.

Local downloaded evidence is under `test_results/kaggle_cayleypy_results_ingest_task4_gate/outputs_v9/`. `npm-gate-results.json` SHA-256 is `74db60b21cbb8656b0f6210c45a2d07a5d26f986643198a7a2e91c0c0d611f3d`; `payload-sha256.json` SHA-256 is `0fcb5e1864fe53d9d9774f1a5fbe58a03cfc4fe68d92e9c412292c15eb9c079e`.
