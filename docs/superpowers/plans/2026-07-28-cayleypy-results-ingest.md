# CayleyPy Results Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy a public Cloudflare ingestion service that accepts validated CayleyPy notebook result envelopes from at least 100 concurrent clients, durably preserves accepted payloads, independently replays solutions, and safely publishes append-only records to `TryDotAtwo/cayleypy-beam-results` through a GitHub App and CI gate.

**Architecture:** A stateless Worker validates and persists each request to R2 and D1 before returning `202`, then enqueues per-result jobs. Queue consumers replay solutions and update an idempotent D1 state machine. A Durable Object serializes/batches GitHub App writes to a staging branch; the results repository independently validates append-only records and deterministic indexes before auto-merge. Raw accepted payloads and exhausted failures remain recoverable in R2/DLQ.

**Tech Stack:** Cloudflare Workers TypeScript, Wrangler, D1, R2, Queues, Durable Objects, Vitest/Miniflare, Ajv JSON Schema, `jose` GitHub App JWT, GitHub Actions, Python 3.12 CI replay/index tools, k6 load testing.

## Global Constraints

- Accept only schema version 1 and reject unknown fields.
- Store the raw request in R2 before returning an accepted receipt.
- Target at least 100 concurrent notebook publishers without synchronous GitHub calls.
- Treat Queue delivery as at-least-once and every state transition as idempotent.
- Never trust a client-selected repository path or filename.
- Independently replay every solution from its bounded proof bundle before staging.
- Preserve accepted raw payloads and exhausted jobs for operator recovery.
- Store GitHub App credentials only in Cloudflare Secrets.
- Use append-only unique result files; derive indexes later.
- Claimed authors are explicitly marked unverified until a future identity layer exists.
- A GitHub outage must not lose accepted results.
- Do not deploy production or publish the Kaggle notebook until 100-client load/recovery, secret, and append-only gates pass.

---

## File Structure

Current repository:

- `services/cayleypy-results-ingest/package.json`: pinned Worker/test dependencies.
- `services/cayleypy-results-ingest/wrangler.jsonc`: staging/production D1, R2, Queue, DO bindings and limits.
- `services/cayleypy-results-ingest/src/schema.ts`: exact TypeScript result types and Ajv validation.
- `services/cayleypy-results-ingest/src/ids.ts`: UUIDv7/idempotency/canonical JSON helpers.
- `services/cayleypy-results-ingest/src/db.ts`: D1 state transitions and queries.
- `services/cayleypy-results-ingest/src/storage.ts`: immutable R2 object writes/reads.
- `services/cayleypy-results-ingest/src/replay.ts`: permutation/path validation.
- `services/cayleypy-results-ingest/src/worker.ts`: `/v1/results`, `/v1/submissions/:id`, health, rate/error responses.
- `services/cayleypy-results-ingest/src/consumer.ts`: validation Queue consumer.
- `services/cayleypy-results-ingest/src/github-app.ts`: installation-token and Git data API client.
- `services/cayleypy-results-ingest/src/github-writer.ts`: Durable Object batch/staging writer.
- `services/cayleypy-results-ingest/migrations/0001_initial.sql`: D1 schema.
- `services/cayleypy-results-ingest/test/`: unit/integration/load fixtures.
- `services/cayleypy-results-ingest/load/k6-100-publishers.js`: concurrency/recovery load test.

Separate `TryDotAtwo/cayleypy-beam-results` checkout:

- `schemas/result-v1.schema.json`: canonical repository schema.
- `tools/validate_result.py`: schema/path replay and placement validation.
- `tools/build_indexes.py`: deterministic derived indexes.
- `.github/workflows/validate-ingest.yml`: append-only validation and index generation.
- `.github/workflows/merge-ingest.yml`: GitHub App staging PR auto-merge gate.

### Task 1: Worker Project, Exact Schema, and Local Runtime

**Files:**
- Create: `services/cayleypy-results-ingest/package.json`
- Create: `services/cayleypy-results-ingest/tsconfig.json`
- Create: `services/cayleypy-results-ingest/wrangler.jsonc`
- Create: `services/cayleypy-results-ingest/src/schema.ts`
- Create: `services/cayleypy-results-ingest/test/schema.test.ts`
- Copy and normalize: `configs/cayleypy_results_schema_v1.json`

**Interfaces:**
- Produces: `validateBatch(value: unknown) -> { ok: true; value: ResultBatch } | { ok: false; errors: SafeSchemaError[] }`.
- Produces exact `ResultEnvelopeV1` and `ResultBatchV1` types shared by Worker/consumer.

- [ ] **Step 1: Create package scripts and pinned dependencies**

```json
{
  "private": true,
  "type": "module",
  "scripts": {
    "test": "vitest run",
    "typecheck": "tsc --noEmit",
    "dev": "wrangler dev --env staging",
    "deploy:staging": "wrangler deploy --env staging",
    "deploy:production": "wrangler deploy --env production"
  },
  "devDependencies": {
    "@cloudflare/vitest-pool-workers": "0.8.55",
    "@cloudflare/workers-types": "4.20260728.0",
    "ajv": "8.17.1",
    "jose": "6.0.11",
    "typescript": "5.8.3",
    "vitest": "3.2.4",
    "wrangler": "4.26.0"
  }
}
```

If an exact package version is unavailable at execution time, resolve it once, record the lockfile-selected version in the test report, and do not use a floating range.

- [ ] **Step 2: Write failing strict-schema tests**

Test a valid result and rejection for unknown field, unsupported schema, oversized author/path/proof, invalid enum, more than 100 results, and serialized request over 25 MiB. Verify errors contain JSON pointer/code only, not submitted values.

- [ ] **Step 3: Run and verify RED**

Run: `npm test -- schema.test.ts`
Expected: FAIL because `src/schema.ts` does not exist.

- [ ] **Step 4: Implement compiled Ajv validation and safe errors**

```ts
export type ValidationResult =
  | { ok: true; value: ResultBatchV1 }
  | { ok: false; errors: Array<{ path: string; keyword: string }> };

export function validateBatch(value: unknown): ValidationResult {
  if (!validate(value)) {
    return { ok: false, errors: (validate.errors ?? []).map(e => ({
      path: e.instancePath, keyword: e.keyword,
    })) };
  }
  return { ok: true, value: value as ResultBatchV1 };
}
```

- [ ] **Step 5: Add staging/production binding names without resource ids**

Declare logical bindings `RESULTS_DB`, `RAW_RESULTS`, `VALIDATE_QUEUE`, `VALIDATE_DLQ`, `GITHUB_WRITER`, and environment variables for repo owner/name/staging branch. Resource ids are injected only after creation and are not invented in source.

- [ ] **Step 6: Test, typecheck, lock, and commit**

Run: `npm install --package-lock-only && npm test && npm run typecheck`
Expected: PASS.

```bash
git add services/cayleypy-results-ingest configs/cayleypy_results_schema_v1.json
git commit -m "feat: scaffold strict CayleyPy results Worker"
```

### Task 2: Idempotent D1 State Machine and Immutable R2 Receipt

**Files:**
- Create: `services/cayleypy-results-ingest/migrations/0001_initial.sql`
- Create: `services/cayleypy-results-ingest/src/ids.ts`
- Create: `services/cayleypy-results-ingest/src/db.ts`
- Create: `services/cayleypy-results-ingest/src/storage.ts`
- Create: `services/cayleypy-results-ingest/test/receipt.test.ts`

**Interfaces:**
- Produces: `canonicalJson(value: unknown) -> string`
- Produces: `computeIdempotency(envelope: ResultEnvelopeV1) -> Promise<string>`
- Produces: `receiveEnvelope(env, envelope, requestMeta) -> Promise<Receipt>`.

- [ ] **Step 1: Write the D1 migration**

```sql
CREATE TABLE submissions (
  submission_id TEXT PRIMARY KEY,
  idempotency_key TEXT NOT NULL UNIQUE,
  run_id TEXT NOT NULL,
  author_name TEXT NOT NULL,
  competition TEXT NOT NULL,
  puzzle_type TEXT NOT NULL,
  puzzle_id INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN (
    'received','queued','validating','validated','rejected','staged','published','retryable','dead_letter'
  )),
  raw_r2_key TEXT NOT NULL UNIQUE,
  safe_error TEXT,
  retry_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  github_path TEXT,
  github_commit_sha TEXT
);
CREATE INDEX submissions_lookup ON submissions(competition,puzzle_type,puzzle_id,created_at);
CREATE INDEX submissions_run ON submissions(run_id,created_at);
```

- [ ] **Step 2: Write failing durability/idempotency tests**

Use Miniflare bindings. Assert R2 `put` occurs before D1 received/queued state and before the returned receipt. Inject Queue failure: raw object and `retryable` row remain. Submit the same idempotency key twice: return the original submission id and enqueue once.

- [ ] **Step 3: Run and verify RED**

Run: `npm test -- receipt.test.ts`
Expected: FAIL with missing modules.

- [ ] **Step 4: Implement canonical ids and immutable R2 keys**

R2 key format is service-generated:

```ts
const rawKey = `raw/v1/${yyyy}/${mm}/${dd}/${submissionId}.json`;
```

Use `If-None-Match: *` semantics where supported; otherwise test existence and treat a different body under the same generated id as terminal corruption. Store SHA-256 in R2 custom metadata.

- [ ] **Step 5: Implement compare-and-transition helpers**

```ts
export async function transition(
  db: D1Database, id: string, from: SubmissionState[], to: SubmissionState,
  patch: { safeError?: string; githubPath?: string; githubCommitSha?: string } = {},
): Promise<boolean>
```

The SQL update includes `WHERE submission_id=? AND state IN (...)`; affected row count must be one or the transition is an idempotent no-op/conflict.

- [ ] **Step 6: Test and commit**

Run: `npm test -- receipt.test.ts && npm run typecheck`
Expected: PASS.

```bash
git add services/cayleypy-results-ingest/migrations services/cayleypy-results-ingest/src/{ids,db,storage}.ts services/cayleypy-results-ingest/test/receipt.test.ts
git commit -m "feat: persist CayleyPy result receipts durably"
```

### Task 3: Public HTTP API, Status, and Abuse Limits

**Files:**
- Create: `services/cayleypy-results-ingest/src/worker.ts`
- Create: `services/cayleypy-results-ingest/test/worker.test.ts`

**Interfaces:**
- Produces routes: `POST /v1/results`, `GET /v1/submissions/:id`, `GET /healthz`.
- `POST` returns `202 { receipts: [{ submission_id, idempotency_key, status_url }] }`.

- [ ] **Step 1: Write failing route tests**

Assert method/content-type/body-size/schema failures; per-IP `429` with `Retry-After`; accepted batch receipts; duplicate receipt; safe `404`; health response without binding/secret detail; emergency `INGEST_MODE=reject|store_only|normal` behavior.

- [ ] **Step 2: Run and verify RED**

Run: `npm test -- worker.test.ts`
Expected: FAIL because `worker.ts` does not exist.

- [ ] **Step 3: Implement bounded body parsing and receipts**

Read `Content-Length` before body, then stream/count with a 25 MiB hard limit. Accept only `application/json`. Persist/enqueue each result independently and return mixed receipt status without echoing result content.

- [ ] **Step 4: Implement rate limiting and emergency modes**

Use a Cloudflare Rate Limiting binding when available plus a D1/global fallback counter. Start with 30 requests/minute/IP, 100 envelopes/request, and 2,000 envelopes/minute globally; expose exact limits in `/healthz` but no infrastructure ids. `store_only` persists to R2/D1 without Queue publication.

- [ ] **Step 5: Test and commit**

Run: `npm test -- worker.test.ts receipt.test.ts && npm run typecheck`
Expected: PASS.

```bash
git add services/cayleypy-results-ingest/src/worker.ts services/cayleypy-results-ingest/test/worker.test.ts
git commit -m "feat: expose bounded public result receipts"
```

### Task 4: Queue Consumer and Independent Permutation Replay

**Files:**
- Create: `services/cayleypy-results-ingest/src/replay.ts`
- Create: `services/cayleypy-results-ingest/src/consumer.ts`
- Create: `services/cayleypy-results-ingest/test/replay.test.ts`
- Create: `services/cayleypy-results-ingest/test/consumer.test.ts`

**Interfaces:**
- Produces: `validateSolution(envelope: ResultEnvelopeV1) -> ReplayResult`.
- Produces Queue handler `queue(batch: MessageBatch<ValidationMessage>, env: Env): Promise<void>`.

- [ ] **Step 1: Write failing replay property tests**

Generate small random permutations and valid paths. Assert exact final-state equality; reject unknown move, non-bijection, wrong state length, class overflow, claimed length mismatch, altered initial/central state, and reflected path whose original-oriented candidate does not solve the original.

- [ ] **Step 2: Write failing duplicate/out-of-order consumer tests**

Deliver the same message twice, deliver after state already `validated`, deliver while `retryable`, and inject malformed R2 body/hash mismatch. Assert one validated transition, terminal rejection with code, or retry/DLQ without duplicate GitHub enqueue.

- [ ] **Step 3: Run and verify RED**

Run: `npm test -- replay.test.ts consumer.test.ts`
Expected: FAIL with missing modules.

- [ ] **Step 4: Implement exact replay with bounded work**

Validate proof permutations before applying moves. Bound `state_len <= 128`, `move_count <= 256`, `path_moves <= 4096`, and total integer cells in the proof. Use typed integer arrays and no dynamic code evaluation.

- [ ] **Step 5: Implement Queue state transitions and retry policy**

Transition `queued|retryable -> validating`; terminal validation errors become `rejected`; R2/D1/DO/network errors use `message.retry({delaySeconds})` with capped exponential delay. After configured attempts, write `dead_letter`, keep raw R2, and enqueue to `VALIDATE_DLQ`.

- [ ] **Step 6: Test and commit**

Run: `npm test -- replay.test.ts consumer.test.ts && npm run typecheck`
Expected: PASS.

```bash
git add services/cayleypy-results-ingest/src/{replay,consumer}.ts services/cayleypy-results-ingest/test/{replay,consumer}.test.ts
git commit -m "feat: replay and validate queued CayleyPy results"
```

### Task 5: GitHub App Client and Serialized Staging Writer

**Files:**
- Create: `services/cayleypy-results-ingest/src/github-app.ts`
- Create: `services/cayleypy-results-ingest/src/github-writer.ts`
- Create: `services/cayleypy-results-ingest/test/github-app.test.ts`
- Create: `services/cayleypy-results-ingest/test/github-writer.test.ts`

**Interfaces:**
- Produces: `getInstallationToken(env, now) -> Promise<string>` with in-memory expiry cache.
- Produces Durable Object RPC: `enqueueValidated(submissionId: string) -> Promise<void>` and alarm-driven `flush() -> Promise<FlushResult>`.

- [ ] **Step 1: Write failing GitHub JWT/token tests**

Mock GitHub endpoints. Assert JWT `iss`, `iat`, `exp <= 10m`, installation-only token request, repository-scoped permissions, cache refresh before expiry, and no private-key/token material in thrown errors/logs.

- [ ] **Step 2: Write failing writer batching/conflict tests**

Queue 100 unique validated ids. Assert a bounded batch creates unique paths, one tree/commit/ref update, and marks D1 rows staged. Inject Git ref `422` conflict twice; writer refetches head and retries without losing/duplicating files. Existing different content at a target path is terminal corruption.

- [ ] **Step 3: Run and verify RED**

Run: `npm test -- github-app.test.ts github-writer.test.ts`
Expected: FAIL with missing modules.

- [ ] **Step 4: Implement secret-only GitHub App authentication**

Bindings/secrets: `GITHUB_APP_ID`, `GITHUB_APP_INSTALLATION_ID`, `GITHUB_APP_PRIVATE_KEY`. Normalize escaped newlines in memory only. Never return secrets from methods or health endpoints.

- [ ] **Step 5: Implement append-only path and batch commit**

```ts
function resultPath(r: ResultEnvelopeV1): string {
  return [
    "results", "v1", safe(r.competition), safe(r.puzzle_type),
    String(r.puzzle_id), r.submitted_at.slice(0, 10), `${r.submission_id}.json`,
  ].join("/");
}
```

Only `safe()`-normalized identifiers from allowlisted fields affect paths. File content is canonical JSON. Batch maximum: 100 records or 5 MiB, alarm within 30 seconds.

- [ ] **Step 6: Test and commit**

Run: `npm test -- github-app.test.ts github-writer.test.ts && npm run typecheck`
Expected: PASS.

```bash
git add services/cayleypy-results-ingest/src/{github-app,github-writer}.ts services/cayleypy-results-ingest/test/{github-app,github-writer}.test.ts
git commit -m "feat: stage validated results through GitHub App"
```

### Task 6: Results Repository Validation, Indexes, and Auto-Merge Gate

**Files in separate `TryDotAtwo/cayleypy-beam-results` checkout:**
- Create: `schemas/result-v1.schema.json`
- Create: `tools/validate_result.py`
- Create: `tools/build_indexes.py`
- Create: `tests/test_validate_result.py`
- Create: `tests/test_build_indexes.py`
- Create: `.github/workflows/validate-ingest.yml`
- Create: `.github/workflows/merge-ingest.yml`

**Interfaces:**
- `python tools/validate_result.py --base "$GITHUB_BASE_SHA" --head "$GITHUB_HEAD_SHA"` validates placement, append-only diff, schema, hashes, and replay.
- `python tools/build_indexes.py --results results --out data` deterministically regenerates TSV/JSON indexes.

- [ ] **Step 1: Clone the results repository into a separate worktree and read its AGENTS.md**

Do not mix its commits with MultiGPUBeamSearch. Create a `codex/results-ingest-v1` branch.

- [ ] **Step 2: Write failing repository validation tests**

Fixtures: valid append; modified/deleted old file; wrong path; duplicate submission/idempotency; malformed proof; invalid solution; executable/workflow content in result; oversized JSON. Assert explicit failure codes.

- [ ] **Step 3: Implement validator**

Use Python `jsonschema`, canonical SHA-256, integer/permutation/path limits matching Worker constants, and exact replay. `git diff --name-status base..head` must contain only `A` records under `results/v1/.../*.json` plus deterministic generated files created by CI itself.

- [ ] **Step 4: Write failing deterministic-index tests**

Assert stable ordering by competition/puzzle type/puzzle id/path length/submission id, author/run indexes, duplicate suppression, and byte-identical output across shuffled filesystem order.

- [ ] **Step 5: Implement derived indexes**

Generate `data/index.tsv`, `data/by_author.tsv`, `data/best_solutions.tsv`, and `data/runs.json`. Never mutate raw result records.

- [ ] **Step 6: Add CI workflows**

`validate-ingest.yml` checks out full history, installs pinned Python dependencies, validates diff, rebuilds indexes, and fails if the regenerated diff is not exactly the expected generated files. `merge-ingest.yml` runs only for the GitHub App staging PR label, requires validation success, and uses GitHub auto-merge; no PAT.

- [ ] **Step 7: Run tests and commit in the results repository**

Run: `python -m pytest tests -q`
Run: `python tools/validate_result.py --base HEAD~1 --head HEAD`
Expected: PASS.

```bash
git add schemas tools tests .github/workflows
git commit -m "feat: validate and index append-only CayleyPy results"
```

### Task 7: Staging Infrastructure and End-to-End Receipt

**Files:**
- Modify: `services/cayleypy-results-ingest/wrangler.jsonc` with real tool-returned staging ids.
- Create: `services/cayleypy-results-ingest/test/staging-smoke.ts`
- Create: `test_results/cayleypy_results_ingest_staging_2026-07-28.md`
- Modify: `memory/CHANGELOG.md`

**Interfaces:**
- Produces a staging URL and one submission that reaches `published` in D1 and the results-repository staging/PR flow.

- [ ] **Step 1: Use the Cloudflare skill and inspect the authenticated account/project state**

Create staging D1, R2, validation Queue, DLQ, Durable Object migration, and Worker only through returned ids. Do not invent resource ids. Keep production separate.

- [ ] **Step 2: Install GitHub App staging secrets through Cloudflare Secrets**

Use an app installation restricted to `TryDotAtwo/cayleypy-beam-results` contents/PR metadata. Confirm no broader organization/repository permission.

- [ ] **Step 3: Apply D1 migration and deploy staging**

Run the exact Wrangler commands selected by the Cloudflare skill. Record deployment id/URL and migration output in the report, excluding secrets.

- [ ] **Step 4: Send one valid and one invalid replay proof**

Assert valid returns `202` and reaches `published`; invalid returns `202` then status `rejected` with a safe validation code. Confirm raw R2 exists for both and GitHub contains only the valid record.

- [ ] **Step 5: Simulate GitHub outage**

Disable GitHub writer access or point staging mock to a controlled `503`. Submit a valid result, confirm accepted/raw/validated/retryable, restore access, and confirm eventual staging without resubmission.

- [ ] **Step 6: Record, test, and commit**

Run: `npm test && npm run typecheck`
Expected: PASS.

```bash
git add services/cayleypy-results-ingest/wrangler.jsonc services/cayleypy-results-ingest/test/staging-smoke.ts test_results/cayleypy_results_ingest_staging_2026-07-28.md memory/CHANGELOG.md
git commit -m "test: validate staged CayleyPy results ingestion"
```

### Task 8: 100-Publisher Load, Duplicate, and Recovery Gate

**Files:**
- Create: `services/cayleypy-results-ingest/load/k6-100-publishers.js`
- Create: `services/cayleypy-results-ingest/test/recovery-audit.ts`
- Create: `test_results/cayleypy_results_ingest_load_100_2026-07-28.md`

**Interfaces:**
- Produces a bounded snapshot proving every accepted receipt is `published`, terminal `rejected`, or recoverably `retryable|dead_letter` with its raw R2 object.

- [ ] **Step 1: Implement deterministic 100-client load data**

Each virtual user sends mixed puzzle ids/authors, 80 unique valid results, 10 exact duplicates, and 10 invalid results. Add bounded jitter and repeat under a controlled GitHub outage window.

- [ ] **Step 2: Set explicit k6 thresholds**

```javascript
export const options = {
  vus: 100,
  iterations: 100,
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<2000"],
    checks: ["rate>0.99"],
  },
};
```

`202`, duplicate `200/202`, expected invalid receipt, and bounded `429` retry are classified explicitly; unexpected `5xx` fails.

- [ ] **Step 3: Run baseline and outage load tests**

Run against staging only. Capture k6 JSON summary, Worker metrics, Queue backlog, D1 counts, R2 counts, DO flush counts, GitHub commit/PR counts, and DLQ.

- [ ] **Step 4: Implement and run recovery audit**

Fetch every receipt id and compare D1/R2/GitHub. Assert no accepted id is missing raw R2, no idempotency key has two repository files, and every validated unique result eventually publishes after outage recovery.

- [ ] **Step 5: Record evidence and commit**

```bash
git add services/cayleypy-results-ingest/load services/cayleypy-results-ingest/test/recovery-audit.ts test_results/cayleypy_results_ingest_load_100_2026-07-28.md
git commit -m "test: prove 100-client results durability"
```

### Task 9: Production Deployment and Operational Controls

**Files:**
- Modify: `services/cayleypy-results-ingest/wrangler.jsonc` with tool-returned production ids.
- Create: `services/cayleypy-results-ingest/OPERATIONS.md`
- Create: `test_results/cayleypy_results_ingest_production_2026-07-28.md`
- Modify: `memory/CHANGELOG.md`

**Interfaces:**
- Produces the stable production `RESULTS_INGEST_URL`, health/status endpoints, emergency mode procedure, replay procedure, and evidence required by the notebook release task.

- [ ] **Step 1: Secret/public-field audit**

Search source, lockfile, Wrangler config, test fixtures, reports, Git diff, and planned Worker responses for private key material, installation tokens, PAT patterns, Cloudflare API tokens, and resource secrets. Any hit blocks deploy.

- [ ] **Step 2: Create isolated production resources through the Cloudflare skill**

Create production D1/R2/Queue/DLQ/DO/Worker separately from staging. Apply migrations. Install GitHub App secrets with repository-only permissions.

- [ ] **Step 3: Deploy production with `INGEST_MODE=store_only`**

Verify health, raw receipt durability, status lookup, and operator visibility without Queue/GitHub publication.

- [ ] **Step 4: Enable normal mode and publish one bounded result**

Change only the environment mode, submit one known valid proof, and verify R2, D1 state history, Queue, DO batch, GitHub staging PR, CI, merge, and final status `published`.

- [ ] **Step 5: Write operations guide**

Document safe commands for health, D1 submission lookup, R2 presence, Queue/DLQ metrics, retry/replay by submission id, `normal|store_only|reject`, GitHub App key rotation, rollback to the previous Worker version, and incident evidence capture. Do not put secrets in commands.

- [ ] **Step 6: Record URL/version/evidence and commit**

```bash
git add services/cayleypy-results-ingest/wrangler.jsonc services/cayleypy-results-ingest/OPERATIONS.md test_results/cayleypy_results_ingest_production_2026-07-28.md memory/CHANGELOG.md
git commit -m "docs: release CayleyPy results ingestion service"
```

- [ ] **Step 7: Hand the production URL to the notebook release plan**

Update only the public notebook's default `RESULTS_INGEST_URL`, regenerate, re-run notebook secret/static tests, and continue with Task 9 of the notebook plan.