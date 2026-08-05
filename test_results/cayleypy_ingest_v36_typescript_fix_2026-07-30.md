# CayleyPy ingest v36 focused TypeScript fix verification

Date: 2026-07-30

Scope: exact private npm gate v36 diagnostics only; no Kaggle rerun, deploy,
resource mutation, or beam/CUDA change.

## RED

- Private gate v36 reported `operator-replay.ts` passing `number | undefined`
  to `Math.min`.
- Private gate v36 reported `deployment-runbook.test.ts` chaining
  `.toBe(false)` onto the matcher result.
- A local focused static contract reproduced the missing explicit `undefined`
  narrowing before the source change.

## GREEN

- Focused source contract: `focused-static-contracts: PASS`
- Staging migration resolver integration test:
  `STAGING_MIGRATIONS_RESOLVER_TEST_OK`
- Patch hygiene: `git diff --check` completed without diagnostics.

The full npm/Vitest/typecheck suite was not rerun locally because the worktree
does not have local TypeScript/Vitest dependencies. Per request, the private
Kaggle gate was not repushed. The fix is limited to the exact v36 diagnostics.
