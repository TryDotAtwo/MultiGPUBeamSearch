# CayleyPy results ingest: fail-closed staging deployment runbook

This runbook provisions and activates **only the `staging` environment** of `services/cayleypy-results-ingest`. It makes no CUDA or beam-search change. It is intentionally two-phase: a working Worker first runs in `store_only`, then a separately reviewed activation deploy changes it to `normal`.

## Non-negotiable safety rules

* Run commands from `services/cayleypy-results-ingest` with the locked `wrangler@4.115.0` from `package-lock.json`; do not use a global Wrangler.
* Use Node `22.23.1`, matching the exact private runtime gate; the recovery audit uses its built-in `--experimental-strip-types` runner and does not depend on an undeclared `tsx` package.
* Do not copy an ID from a dashboard guess or another environment. The D1 UUID passed to the helper must be copied from the current `wrangler d1 create` response (or the matching current-account `wrangler d1 list` response).
* The deployment helper writes a temporary generated config. It never edits the tracked `wrangler.jsonc` and never stores a GitHub credential.
* The GitHub App bootstrap accepts only the live staging Worker origin, requires `/healthz` to report `store_only`, and uses the already-generated private config at `.staging-deploy-private/wrangler.generated.json`.
* Never put a GitHub App private key in a command line, config file, environment variable, clipboard, log, k6 variable, or git commit. The bootstrap keeps GitHub's PKCS#1 response in memory, converts it to PKCS#8 in memory, and pipes one JSON object directly to pinned Wrangler `secret bulk` stdin.
* The bootstrap creates a private personal-account App only after the operator passes the explicit live confirmation flag. Its dry-run performs read-only checks and creates no App, listener, browser flow, resource, version, key, or secret.
* `normal` is forbidden until a store-only POST/status/R2/D1 audit has passed. If any gate fails, keep or return to `store_only`; do not retry activation.

## Prerequisites and exact resource inventory

Authenticate only after confirming the intended Cloudflare account. The commands below create nothing until they are explicitly run; record their sanitized tool output in the private deployment evidence file.

```powershell
Set-Location services/cayleypy-results-ingest
& .\node_modules\.bin\wrangler.cmd whoami --json
if ($LASTEXITCODE -ne 0) { throw 'wrangler whoami failed' }
$env:CLOUDFLARE_ACCOUNT_ID = 'paste one selected 32-hex account id here'
& .\node_modules\.bin\wrangler.cmd whoami --account $env:CLOUDFLARE_ACCOUNT_ID --json
if ($LASTEXITCODE -ne 0) { throw 'account-pinned wrangler whoami failed' }

# Copy the same selected account id into the private manifest. The canonical
# CLOUDFLARE_ACCOUNT_ID pins every resource command below to that exact account.
# Create once in the selected account. Copy the D1 database_id returned by the
# first command into the private manifest; do not invent one. Every native CLI
# call is checked immediately so Windows PowerShell 5.1 cannot continue after a
# failed authentication or resource mutation.
& .\node_modules\.bin\wrangler.cmd d1 create cayleypy-results-staging
if ($LASTEXITCODE -ne 0) { throw 'D1 creation failed' }
& .\node_modules\.bin\wrangler.cmd r2 bucket create cayleypy-results-raw-staging
if ($LASTEXITCODE -ne 0) { throw 'R2 creation failed' }
& .\node_modules\.bin\wrangler.cmd queues create cayleypy-validate-staging
if ($LASTEXITCODE -ne 0) { throw 'validation Queue creation failed' }
& .\node_modules\.bin\wrangler.cmd queues create cayleypy-validate-dlq-staging
if ($LASTEXITCODE -ne 0) { throw 'DLQ creation failed' }
```

The SQLite-backed `GitHubWriter` Durable Object is not separately created: the existing declarative `exports.GitHubWriter` block provisions it on the first `wrangler deploy --env staging`. Do not add legacy `migrations` to this Worker; `exports` and legacy DO migrations are mutually exclusive.

Create a private local manifest (it is deliberately ignored by git):

```powershell
@'
{
  "account_label": "copied from wrangler whoami output",
  "cloudflare_account_id": "paste the selected account's exact 32-hex id here",
  "d1_database_name": "cayleypy-results-staging",
  "d1_database_id": "paste only the current wrangler-returned database_id here",
  "r2_bucket_name": "cayleypy-results-raw-staging",
  "validate_queue_name": "cayleypy-validate-staging",
  "validate_dlq_name": "cayleypy-validate-dlq-staging"
}
'@ | Set-Content -LiteralPath .\staging-resources.private.json -Encoding UTF8 -NoNewline
```

The helper rejects placeholder values, foreign names, a missing or malformed 32-hex Cloudflare account id, non-UUID D1 ids, missing files, environment/account conflicts, and any manifest outside this exact resource set.

## Preflight and `store_only` deployment

```powershell
& .\scripts\invoke-staging-deployment.ps1 -Phase preflight -ResourceManifest .\staging-resources.private.json

# Applies only deployable D1 SQL migrations in sorted order before the Worker.
& .\scripts\invoke-staging-deployment.ps1 -Phase store_only -ResourceManifest .\staging-resources.private.json
```

The helper performs: local pinned-Wrangler version/config checks, verifies membership in the exact manifest account with `wrangler whoami --account <account-id> --json`, renders a temporary config with that top-level `account_id`, the exact supplied D1 id, and an absolute `migrations_dir` resolving to this service's tracked `migrations/`, applies every pending migration in Wrangler order (currently `0001_initial.sql`, `0002_ingest_rate_limits.sql`, then `0003_remove_legacy_status_ip_limits.sql`) with `wrangler d1 migrations apply ... --remote`, and deploys it with `INGEST_MODE=store_only`. A conflicting `CLOUDFLARE_ACCOUNT_ID` fails closed. It does not manufacture an endpoint: read the Worker URL only from the successful `wrangler deploy` output and place it in the private shell session for the following checks.

## One-shot private GitHub App bootstrap

Install GitHub CLI and authenticate it as the personal account `TryDotAtwo` before continuing. The existing authenticated `gh` session is used only for read-only identity and repository-admin preflight checks; its OAuth token cannot enumerate GitHub App installations and is never reused as App authentication. No PAT is requested or stored. The fixed App contract is:

* owner account: personal `TryDotAtwo` (`User`), private / only on this account;
* App name: `cayleypy-beam-results-ingest`;
* repository permission: `contents=write`, implicit `metadata=read`, and no other permission;
* webhooks inactive, no events, no OAuth callback, no setup URL;
* installation: **Only select repositories**, then only `TryDotAtwo/cayleypy-beam-results` (repository id `1281329788`).

First run the mutation-free dry-run against the actual staging Worker origin copied from the successful store-only deployment:

```powershell
$env:CAYLEYPY_STAGING_ENDPOINT = 'https://the-actual-staging-worker-origin.example'
& npm.cmd run bootstrap:github-app:dry-run -- --staging-endpoint $env:CAYLEYPY_STAGING_ENDPOINT
```

The dry-run fails closed unless all of these read-only gates pass:

1. the exact local `wrangler@4.115.0` package and entrypoint exist;
2. the fixed generated config is a regular file, pins the exact lowercase 32-hex Cloudflare `account_id`, contains the exact staging resource inventory and D1 UUID, contains no secret, and resolves `INGEST_MODE=store_only`;
3. `wrangler check`, account-pinned `wrangler whoami --json`, `gh auth status`, exact `TryDotAtwo` identity, exact repository id/admin access, and live `/healthz` `store_only` all succeed;
4. `wrangler secret list --format json` reports an empty array for that exact staging Worker before any App, listener, browser, token, or secret mutation. Any pre-existing secret is a manual-review stop because `secret bulk` preserves omitted names.

Run the live bootstrap under a single reviewed operator with concurrent staging-secret administration paused. The helper repeats the empty-list check immediately before `secret bulk` and verifies the exact three-name/type set afterward, but Cloudflare exposes no atomic secret-list compare-and-set: a concurrent administrator could still race between those calls. If the post-check fails, keep `store_only`, stop, and inspect the created Worker version and full secret inventory; do not retry blindly.

After reviewing `DRY_RUN_OK`, start the one permitted live flow:

```powershell
& npm.cmd run bootstrap:github-app -- --staging-endpoint $env:CAYLEYPY_STAGING_ENDPOINT --confirm-create-private-app
```

The helper binds an ephemeral listener only to `127.0.0.1`, generates 32 random state bytes, serves one fixed personal-account manifest form, and accepts exactly one matching `GET /github-app-manifest/callback` containing one code and one state. It never prints the callback query or conversion response. Complete both opened browser pages without changing the App contract:

1. create the private App from the fixed manifest;
2. install it with **Only select repositories** and the single required repository.

The helper signs a short-lived GitHub App JWT from the in-memory PKCS#8 key, verifies that the App has exactly one unsuspended personal-account installation, and confirms the required repository resolves to that same installation. It then creates a temporary installation token narrowed to `contents=read` and `metadata=read`, enumerates the installation's complete repository selection, requires exactly repository id `1281329788`, and revokes that token successfully before continuing. It fails closed on an organization, all repositories, another or additional repository, extra permission/event, pagination, suspension, token-scope drift, or revocation failure. Neither JWT nor token is logged or written. Only after that verification does it repeat every store-only prerequisite and run one equivalent command internally:

```text
node node_modules/wrangler/bin/wrangler.js secret bulk --config .staging-deploy-private/wrangler.generated.json --env staging
```

The stdin object has exactly `GITHUB_APP_ID`, `GITHUB_APP_INSTALLATION_ID`, and the in-memory PKCS#8 `GITHUB_APP_PRIVATE_KEY`. Neither the object nor either private-key representation is written to disk or printed. Wrangler creates one store-only Worker version for the bulk operation; the helper then requires exactly those three names and `secret_text` types from `secret list`, with no fourth or missing entry.

At this stage the App secrets must remain unused because `store_only` forbids Queue and GitHub publication. Use one canonical valid envelope, then verify:

1. `/healthz` succeeds and reports the expected safe mode;
2. POST receipt is durable and `/v1/submissions/{submission_id}` is `received`;
3. exact immutable raw object metadata and one D1 row exist;
4. Queue metrics show no sent message and GitHub has no new staging commit.

Store only safe receipt ids, counts, SHA-256 values, timestamps, migration names and Worker version in `test_results/`; do not store payloads or secrets.

## Controlled activation and live load/recovery gate

After a separate reviewer signs the four store-only facts, activate normal mode. This is a config-only redeploy; it does not change the beam solver.

```powershell
& .\scripts\invoke-staging-deployment.ps1 -Phase activate_normal -ResourceManifest .\staging-resources.private.json
if ($LASTEXITCODE -ne 0) { throw 'normal-mode activation failed' }

$env:INGEST_BASE_URL = '<copied from this successful deploy output>'
$env:LOAD_PHASE = 'baseline'
$env:RECOVERY_TIMEOUT_SECONDS = '600'
$env:RECOVERY_POLL_SECONDS = '5'
$evidenceRoot = Resolve-Path ..\..\test_results
$evidenceDirectory = Join-Path $evidenceRoot ('cayleypy-results-ingest-staging-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $evidenceDirectory -ErrorAction Stop | Out-Null
$receiptManifest = Join-Path $evidenceDirectory 'load-normal.log'
$loadSummary = Join-Path $evidenceDirectory 'load-normal-summary.json'
$d1Snapshot = Join-Path $evidenceDirectory 'd1-audit.json'
$r2Snapshot = Join-Path $evidenceDirectory 'r2-audit.json'
$githubSnapshot = Join-Path $evidenceDirectory 'github-audit.json'

k6 run --log-format json --summary-export $loadSummary .\load\k6-100-publishers.js 2>&1 |
  Tee-Object -LiteralPath $receiptManifest
if ($LASTEXITCODE -ne 0) { throw 'k6 staging load failed' }

# Before the audit, populate the three paths below with bounded, sanitized,
# read-only snapshots from this exact staging account/repository. Their required
# fields are listed immediately below this command block. Do not export payloads.
foreach ($snapshot in @($d1Snapshot, $r2Snapshot, $githubSnapshot)) {
  if (-not (Test-Path -LiteralPath $snapshot -PathType Leaf)) {
    throw "missing sanitized audit snapshot: $snapshot"
  }
}

node --experimental-strip-types .\test\recovery-audit.ts `
  --manifest $receiptManifest `
  --d1 $d1Snapshot `
  --r2 $r2Snapshot `
  --github $githubSnapshot
if ($LASTEXITCODE -ne 0) { throw 'strict recovery audit failed' }
```

Populate only these bounded audit views after the load finishes:

* D1: JSON rows or Wrangler JSON pages containing only `submission_id,idempotency_key,state,raw_r2_key,safe_error,github_path`;
* R2: a JSON string-key array or `{ "objects": [{ "key": "..." }] }` inventory;
* GitHub: a JSON array or `{ "entries": [...] }` containing only `path,submission_id,idempotency_key` from the result records.

Generate them with read-only, account-pinned queries against the exact staging resources and repository. The audit rejects oversized or over-count snapshots; never place raw envelopes, response bodies, credentials, private-key bytes, or bearer tokens in these files.

Use only the existing deterministic workload: 80 unique valid envelopes, 10 semantic transport duplicates, and 10 invalid-proof envelopes. Its recovery audit must show 80 unique durable accepted submissions, 10 duplicates, and no loss or duplicate GitHub record. Keep logs receipt-only. A `429` retry remains bounded by the existing script; broad retries are a gate failure.

## Rollback and recovery

For any validation, GitHub, queue, rate-limit, or audit anomaly immediately return to `store_only`, preserve raw objects/D1 evidence, and stop publishers:

```powershell
& .\scripts\invoke-staging-deployment.ps1 -Phase rollback_store_only -ResourceManifest .\staging-resources.private.json
```

Do **not** delete R2 objects, D1 rows, Queue messages, or the Durable Object to "clean up" an incident. Capture the deployed Worker version and use the recovery audit after the fault is understood. `wrangler rollback` is only safe for a version known to contain the same declarative Durable Object lifecycle; do not roll back across its first `exports` provisioning deployment.

If the manifest callback has completed but installation or bulk upload fails, do not rerun the live manifest flow blindly: the App may already exist while its one-time returned key has been discarded. Keep staging in `store_only`, inspect the App and installation in GitHub settings, and either delete the incomplete App before a fresh reviewed bootstrap or generate/rotate a new App key through a separately reviewed in-memory recovery procedure. Revoke any superseded key. Never download a replacement key to the repository or a shared filesystem.

## Current local blocker

This repository previously observed `api.cloudflare.com` timeout / unauthenticated Wrangler access locally, and the local exact dependency tree may not yet contain the Wrangler entrypoint. These are deployment-environment prerequisites, not reasons to substitute guessed account/resource IDs or a global CLI. Record the exact sanitized failure in a private `test_results/` note and stop before resource creation or App creation if dependency restoration, `wrangler whoami`, `gh` authentication, or live store-only health fails.

## GitHub-authoritative continuous deployment

After the initial store-only audit and explicit activation, the live staging Worker is deployed by Cloudflare Workers Builds from `TryDotAtwo/MultiGPUBeamSearch`.

- Root directory: `/services/cayleypy-results-ingest`
- Production branch: `codex/cayleypy-results-ingest` until the branch is merged to `main`
- Build command: `npm ci && npm run ci:cloudflare`
- Deploy command: `npm run deploy:staging:github`
- Deploy config: `wrangler.github-staging.jsonc`
- Include watch path: `services/cayleypy-results-ingest/**`

The tracked Git-build config contains only public resource identifiers and normal-mode variables. Runtime GitHub App credentials remain encrypted Worker secrets and are never copied into the repository or GitHub Actions. Cloudflare generates and retains the scoped Workers Builds token. Manual local deploy remains an emergency rollback path, not the normal release path.