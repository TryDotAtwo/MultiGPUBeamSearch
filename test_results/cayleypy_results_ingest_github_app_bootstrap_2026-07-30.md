# CayleyPy results ingest GitHub App bootstrap preparation ? 2026-07-30

## Outcome

Prepared a fail-closed, one-shot private GitHub App bootstrap for the staging
results-ingest Worker. This preparation performed no live GitHub App creation,
App installation, private-key generation by GitHub, Cloudflare resource/secret
mutation, Worker version creation, deployment, browser flow, external load, or
beam/CUDA change.

The fixed contract is the personal account `TryDotAtwo`, private App
`cayleypy-beam-results-ingest`, only repository
`TryDotAtwo/cayleypy-beam-results` (id `1281329788`), App permissions
`contents=write` and `metadata=read`, no events, and staging only.

## Implemented safety gates

- mutation-free dry-run before the explicit live confirmation flag;
- IPv4 loopback-only one-shot manifest callback with 32 random state bytes,
  exact callback query validation, and redacted failures;
- in-memory GitHub PKCS#1 to Worker PKCS#8 conversion and native short-lived
  App JWT signing;
- exact personal-account installation, selected repository, permission, event,
  suspension, pagination, token-expiry, and repository-cardinality checks;
- temporary installation token narrowed to read-only and always revoked;
- exact 32-hex Cloudflare account in the private manifest and generated config,
  canonical case-insensitive Windows environment handling, and account-pinned
  Wrangler calls;
- no pre-existing staging secrets, a second immediate pre-bulk check, one
  pinned Wrangler `secret bulk` JSON stdin operation, and exact post-upload
  verification of three `secret_text` names;
- BOM-free generated UTF-8 config;
- corrected staging runbook using `INGEST_BASE_URL`, all four required bounded
  recovery-audit inputs, and immediate `$LASTEXITCODE` checks after both
  `whoami` calls and each resource creation command.

Cloudflare does not expose an atomic secret-list/secret-bulk compare-and-set.
The runbook therefore preserves a documented single-operator boundary, repeats
the empty secret inventory check immediately before bulk, and fails closed on
post-check drift.

## Verification completed locally

- `node --check` passed for both bootstrap MJS files.
- Bootstrap `--help` passed and printed only the safe usage contract.
- Node 22 experimental type-stripping syntax checks passed for the declaration,
  bootstrap test, runbook test, recovery audit, and both Vitest configs.
- PowerShell staging migration/config resolver passed, including account pinning,
  sorted migrations, and BOM-free generated JSON.
- Dependency-free adversarial bootstrap smoke passed for callback/state, key
  conversion/JWT claims, generated config, mixed-case Cloudflare environment,
  exact secret inventory/bulk stdin, dry-run non-mutation, installation token
  verification/revocation, and the real loopback listener.
- Dependency-free negative GitHub flow smoke passed for App pagination,
  repository pagination with revoke, invalid far-future expiry with revoke, and
  revoke failure.
- Focused runbook RED/GREEN contract passed for `INGEST_BASE_URL`, four audit
  arguments, removed unused variable, and all six native CLI guards.
- Independent security review and final test re-review report no remaining
  P0-P2 findings. The only noted residual is the documented Cloudflare
  list/bulk TOCTOU boundary.
- Private Kaggle npm-gate builder compiled and rebuilt successfully; metadata is
  private, CPU-only, Internet-enabled, and the embedded payload includes all
  bootstrap/runbook sources and tests.
- `git diff HEAD --check`, beam/CUDA scope scan, private-artifact path scan,
  and high-entropy credential scan passed. PEM markers and secret names in the
  diff are deliberate validation/test literals, not credentials.

## Pending exact dependency gate

The local partial `node_modules` tree still lacks the locked Vitest,
TypeScript, and Wrangler entrypoints. Per controller instruction, this task did
not run `npm ci`. Therefore the full locked `npm test` and
`npm run typecheck` gates remain pending until the exact dependency tree is
restored by the controller or the rebuilt private Kaggle npm gate is run.
No claim of those two gates is made here.
