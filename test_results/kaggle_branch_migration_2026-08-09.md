# Kaggle Branch Migration Verification

## Preserved source

- Source branch: `codex/profile-capacity-safety`
- Source and initial canonical tip: `a0326407ecc58a0b800f34d217a126ce876a7640`
- Canonical branch: `kaggle/notebooks`
- `codex/cube4-public-collector` and `codex/stream1-piece-transformer` are ancestors of the selected tip.
- Source history was not rewritten.

## Static notebook gates

- Tracked notebooks parsed as JSON: `21`
- Tracked Kaggle/metadata JSON files parsed: `16`
- Python code cells compiled: `88`
- Non-Python magic cells skipped: `0`
- Syntax failures: `0`

## Public notebook suite

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q tests/cayleypy_public
```

Result: `276 passed in 21.38s`, exit code `0`.

## Dirty open-PR work preservation

The dirty `codex/public-cayleypy-notebook` worktree was not merged into the
verified canonical branch.

- Binary patch: `test_results/dirty_public_notebook_patch_2026-08-09.patch`
- Patch SHA-256: `d212c123ea9b0f060f81526f4b18bf08c5947b9f98053e5782d823c35f8dae1b`
- External output archive: `D:/100XH100-backups/dirty-public-notebook-outputs-2026-08-09.zip`
- Archive size: `99,087,050` bytes
- Archive SHA-256: `f16fad6d4bb790c53a8385a39521f23473dee2dc72774bb2811a0d0b4d37f8fd`

The local-only commit `38310f89` is independently preserved by legacy refs
and the verified full Git bundle. No Kaggle kernel was pushed or rerun during
this migration gate; local tests do not claim new T4/H100 acceptance.
