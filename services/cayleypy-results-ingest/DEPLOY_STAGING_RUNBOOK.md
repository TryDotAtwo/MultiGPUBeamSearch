# CayleyPy results ingest: fail-closed staging deployment runbook

This runbook provisions and activates **only the `staging` environment** of `services/cayleypy-results-ingest`. It makes no CUDA or beam-search change. It is intentionally two-phase: a working Worker first runs in `store_only`, then a separately reviewed activation deploy changes it to `normal`.

## Non-negotiable safety rules

* Run commands from `services/cayleypy-results-ingest` with the locked `wrangler@4.115.0` from `package-lock.json`; do not use a global Wrangler.
* Do not copy an ID from a dashboard guess or another environment. The D1 UUID passed to the helper must be copied from the current `wrangler d1 create` response (or the matching current-account `wrangler d1 list` response).
* The helper writes a temporary generated config. It never edits the tracked `wrangler.jsonc`, never accepts a URL, and never stores a GitHub credential.
* Never put a GitHub App private key in a command line, config file, log, k6 variable, or git commit. `wrangler secret put` reads it interactively.
* `normal` is forbidden until a store-only POST/status/R2/D1 audit has passed. If any gate fails, keep or return to `store_only`; do not retry activation.

## Prerequisites and exact resource inventory

Authenticate only after confirming the intended Cloudflare account. The commands below create nothing until they are explicitly run; record their sanitized tool output in the private deployment evidence file.

```powershell
Set-Location services/cayleypy-results-ingest
& .\node_modules\.bin\wrangler.cmd whoami

# Create once in the selected account. Copy the D1 database_id returned by the
# first command into the private manifest; do not invent one.
& .\node_modules\.bin\wrangler.cmd d1 create cayleypy-results-staging
& .\node_modules\.bin\wrangler.cmd r2 bucket create cayleypy-results-raw-staging
& .\node_modules\.bin\wrangler.cmd queues create cayleypy-validate-staging
& .\node_modules\.bin\wrangler.cmd queues create cayleypy-validate-dlq-staging
```

The SQLite-backed `GitHubWriter` Durable Object is not separately created: the existing declarative `exports.GitHubWriter` block provisions it on the first `wrangler deploy --env staging`. Do not add legacy `migrations` to this Worker; `exports` and legacy DO migrations are mutually exclusive.

Create a private local manifest (it is deliberately ignored by git):

```powershell
@'
{
  "account_label": "copied from wrangler whoami output",
  "d1_database_name": "cayleypy-results-staging",
  "d1_database_id": "paste only the current wrangler-returned database_id here",
  "r2_bucket_name": "cayleypy-results-raw-staging",
  "validate_queue_name": "cayleypy-validate-staging",
  "validate_dlq_name": "cayleypy-validate-dlq-staging"
}
'@ | Set-Content -NoNewline .\staging-resources.private.json
```

The helper rejects placeholder values, foreign names, non-UUID D1 ids, missing files, and a manifest outside this exact resource set.

## Preflight and `store_only` deployment

```powershell
& .\scripts\invoke-staging-deployment.ps1 -Phase preflight -ResourceManifest .\staging-resources.private.json

# Applies only deployable D1 SQL migrations in sorted order before the Worker.
& .\scripts\invoke-staging-deployment.ps1 -Phase store_only -ResourceManifest .\staging-resources.private.json
```

The helper performs: local pinned-Wrangler version/config checks, renders a temporary config with the exact supplied D1 id and an absolute `migrations_dir` resolving to this service's tracked `migrations/`, applies `0001_initial.sql` and `0002_ingest_rate_limits.sql` with `wrangler d1 migrations apply ... --remote`, and deploys it with `INGEST_MODE=store_only`. It does not manufacture an endpoint: read the Worker URL only from the successful `wrangler deploy` output and place it in the private shell session for the following checks.

Install the three GitHub App values only after the store-only deploy is healthy. The values must be created for the staging named environment and are never printed by this runbook:

```powershell
& .\node_modules\.bin\wrangler.cmd secret put GITHUB_APP_ID --env staging
& .\node_modules\.bin\wrangler.cmd secret put GITHUB_APP_INSTALLATION_ID --env staging
& .\node_modules\.bin\wrangler.cmd secret put GITHUB_APP_PRIVATE_KEY --env staging
& .\node_modules\.bin\wrangler.cmd secret list --env staging
```

At this stage the App secrets must remain unused because `store_only` forbids Queue and GitHub publication. Use one canonical valid envelope, then verify:

1. `/health` succeeds and reports the expected safe mode;
2. POST receipt is durable and `/v1/status/{submission_id}` is `received`;
3. exact immutable raw object metadata and one D1 row exist;
4. Queue metrics show no sent message and GitHub has no new staging commit.

Store only safe receipt ids, counts, SHA-256 values, timestamps, migration names and Worker version in `test_results/`; do not store payloads or secrets.

## Controlled activation and live load/recovery gate

After a separate reviewer signs the four store-only facts, activate normal mode. This is a config-only redeploy; it does not change the beam solver.

```powershell
& .\scripts\invoke-staging-deployment.ps1 -Phase activate_normal -ResourceManifest .\staging-resources.private.json

$env:INGEST_ENDPOINT = '<copied from this successful deploy output>'
$env:RECOVERY_AUDIT_EXPECTED_MODE = 'normal'
k6 run .\load\k6-100-publishers.js
node --import tsx .\test\recovery-audit.ts
```

Use only the existing deterministic workload: 80 unique valid envelopes, 10 semantic transport duplicates, and 10 invalid-proof envelopes. Its recovery audit must show 80 unique durable accepted submissions, 10 duplicates, and no loss or duplicate GitHub record. Keep logs receipt-only. A `429` retry remains bounded by the existing script; broad retries are a gate failure.

## Rollback and recovery

For any validation, GitHub, queue, rate-limit, or audit anomaly immediately return to `store_only`, preserve raw objects/D1 evidence, and stop publishers:

```powershell
& .\scripts\invoke-staging-deployment.ps1 -Phase rollback_store_only -ResourceManifest .\staging-resources.private.json
```

Do **not** delete R2 objects, D1 rows, Queue messages, or the Durable Object to "clean up" an incident. Capture the deployed Worker version and use the recovery audit after the fault is understood. `wrangler rollback` is only safe for a version known to contain the same declarative Durable Object lifecycle; do not roll back across its first `exports` provisioning deployment.

## Current local blocker

This repository previously observed `api.cloudflare.com` timeout / unauthenticated Wrangler access locally. That is a deployment-environment prerequisite, not a reason to substitute guessed account/resource IDs. Record the exact sanitized failure in a private `test_results/` note and stop before resource creation if `wrangler whoami` or any control-plane command fails.
