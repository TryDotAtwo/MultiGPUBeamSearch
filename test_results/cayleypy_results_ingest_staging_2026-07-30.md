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
- Worker version: `12d5ebaa-1472-427a-aabc-c1ca9bebdc82`
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

## Current external gate

The one permitted live GitHub App bootstrap reached GitHub sudo/2FA. App creation has not completed, and the Worker still has no secrets. No result publication or normal-mode activation occurred. The live flow remains fail-closed in `store_only`.

No beam-search or CUDA implementation was changed.
