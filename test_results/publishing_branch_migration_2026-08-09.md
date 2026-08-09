# Publishing Branch Migration Verification

## Preserved source

- Source branch: `codex/cayleypy-results-ingest`
- Source and initial canonical tip: `638fcc90a349f68e01c14dfdca299a87700dadf1`
- Canonical branch: `publishing/results`
- The branch was created directly at the published PR #1 tip; source history was not rewritten.

## Dependency gate

`npm.cmd ci` installed 90 packages from the committed lockfile. NPM reported
7 audit findings (4 moderate, 3 high). No automatic or breaking audit fix was
applied during branch migration.

## Red/green fixture repair

The committed exact-4-MiB streaming test expected a four-byte UTF-8 emoji but
its source fixture contained mojibake bytes and therefore no `0xf0` byte.

- RED: isolated test failed at `expect(emojiStart).toBeGreaterThan(0)`.
- Evidence: committed fixture bytes were `d1 80 d1 9f c2 98 d0 82`.
- Minimal fix: represent the intended character as ASCII source escape `\u{1F600}`.
- GREEN: isolated test passed; 1 passed and 54 skipped.
- Worker implementation was not changed.

## Full local service gate

Command: `npm.cmd run ci:cloudflare`

- Schema/config tests: `72 passed`.
- Worker/service tests: `139 passed`.
- TypeScript: `tsc --noEmit` passed.
- Exit code: `0`.
- No deploy, live D1 migration, Cloudflare mutation, or GitHub publication occurred.

## Publisher plugin gate

Command:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q plugins/cayleypy-results-publisher/tests/test_cayleypy_submit.py
```

Result: `18 passed in 0.29s`, exit code `0`.

## Dirty old-worktree preservation

- Patch: `test_results/dirty_results_ingest_package_patch_2026-08-09.patch`
- Patch SHA-256: `d2c69ac507e93a0729c0f1afdac43036c25fd7e98cbfdce3750de3717981588f`
- The patch contains only dirty `package.json` and `package-lock.json` changes.
- Existing `node_modules` remains in its original worktree and is recorded by the repository SHA manifest; generated dependencies are not merged into Git.
