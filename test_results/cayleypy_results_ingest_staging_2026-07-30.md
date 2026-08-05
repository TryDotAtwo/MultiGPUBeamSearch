# CayleyPy results ingest staging evidence

Date: 2026-07-30

## Provisioned resources

- D1: `cayleypy-results-staging`, region `WEUR`
- R2: `cayleypy-results-raw-staging`
- Queue: `cayleypy-validate-staging`
- DLQ: `cayleypy-validate-dlq-staging`
- Worker: `cayleypy-results-ingest-staging`
- Endpoint: `https://cayleypy-results-ingest-staging.tupa-expert.workers.dev`

All resources were first listed in the exact authenticated account; none of the four target names existed before creation.

## Store-only deployment

- Pinned Wrangler: `4.115.0`
- D1 migrations applied in order: `0001_initial.sql`, `0002_ingest_rate_limits.sql`, `0003_remove_legacy_status_ip_limits.sql`
- Declarative SQLite Durable Object export created: `GitHubWriter`
- Audited Worker version: `374e11ac-3716-414f-a43b-8278010b48cd`
- Live `/healthz`: HTTP 200, `status=ok`, `ingest_mode=store_only`
- Pre-bootstrap secret inventory: exact empty JSON array

## Defects found and fixed

1. Moving the generated Wrangler config into `.staging-deploy-private` changed the resolution base for `main: src/worker.ts`. The helper now resolves and validates an absolute entry point before writing the generated config.
2. Wrangler `4.115.0` treats `wrangler check` as a command group and returned help with exit code 0. The helper now performs a real `wrangler deploy --dry-run --outdir ...` build for both tracked and generated configs.
3. Node fetch did not automatically use the required proxy. The GitHub App bootstrap now installs `EnvHttpProxyAgent` only when a standard proxy environment variable is present.

## Verification

- Both tracked and generated configs passed real Wrangler dry-run builds with the exact staging bindings.
- Re-running migrations reported `No migrations to apply`.
- Full local schema/config suite: 6 files, 55 tests passed.
- Full Worker suite: 6 files, 122 tests passed.
- TypeScript typecheck passed.
- GitHub App bootstrap dry-run passed every exact read-only prerequisite and reported `DRY_RUN_OK`.

## Live completion gate

- GitHub App `4436526`, installation `150104325`, is scoped to exactly `TryDotAtwo/cayleypy-beam-results`; the Worker contains exactly the three required App secrets.
- Store-only receipt `019fb398-8a19-7916-8da0-cc635a2fd5d4` persisted one D1 `received` row and an immutable 2413-byte R2 object. Its SHA-256 `098def251326d17c0f0e573f5ce330fb1f0f79eba8ba271487d7ec928e189316` exactly matched the canonical golden envelope, with no GitHub ref created.
- Normal Worker version `4ca1ba50-eeb4-4be5-85a9-9d3f1d4e52a2` recovered that row across the deployment boundary and staged it in GitHub.
- Free portable k6 v2.1.0 ran 100 VUs: 80 unique valid envelopes, 10 semantic duplicates, and 10 invalid proofs. Five client requests timed out in the local proxy; successful-response p95 was 1.61 seconds and every invalid proof was rejected without a receipt.
- A bounded D1 audit found 78/80 unique rows after the first pass. The two absent requests (`puzzle_id=10021,10053`) were replayed once through the same public endpoint.
- Final D1 state: 80 rows, 80 distinct idempotency keys, all 80 `staged`, zero errors. Final GitHub head: `e4fed45af87d3694010a3c6a7a6b1ea06b09dace`.
- Final GitHub recursive tree: 81 unique `results/v1/**` files (canary plus 80 unique load records), not truncated.
- Raw load evidence: `test_results/cayleypy_results_ingest_k6_baseline_2026-07-30.log`.

The k6 threshold exit remains recorded as a local transport failure; the final bounded D1/GitHub audit is the losslessness evidence.

No beam-search or CUDA implementation was changed.
