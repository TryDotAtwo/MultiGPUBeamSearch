# Public CayleyPy notebook builder — initial verification

Scope: the new thin notebook layer only. It delegates all model export, profile
selection, build, beam-search, validation and result publishing to the existing
`tools/run_cayleypy_public.py` CLI; it does not alter CUDA/C++ beam-search code.

Expected verification command:

```powershell
py -m pytest tests/cayleypy_public/test_public_notebook_builder.py -q
```

The test checks deterministic regeneration, valid nbformat JSON, six ordered
cells, AST parsing of every code cell, clean execution outputs, private 2×T4
Kaggle metadata, the public contract, absence of forbidden user model controls
and secrets, and explicit Kaggle input placeholders. GPU execution and Kaggle
push are intentionally outside this builder-only gate.

Observed 2026-07-29: python -m pytest tests/cayleypy_public/test_public_notebook_builder.py -q from the public worktree passed (3 passed); builder output passed JSON parsing plus AST parsing for every code cell, with no execution outputs. python -m compileall -q tools/build_kaggle_cayleypy_public_notebook.py and git diff --check also passed. This is a structural/offline gate only; it intentionally does not execute Kaggle GPU work or push a kernel.

Review-fix update: reader-facing sources now use ASCII-only `--` and `2xT4` text; the checkout URL is a hard-coded official repository rather than a user setting; `SOLVER_COMMIT` is pinned to `3bbbe50a9460695507aee58bf443c1d3b0bd5032`; the run cell retains a nonzero CLI return code, then the artifact/status cell displays retained outputs and raises afterwards. The focused test suite now also verifies these properties and ASCII-only notebook emission.

Independent final gate after review fixes:
- `py -m pytest tests/cayleypy_public/test_public_notebook_builder.py -q`: 5 passed.
- `py -m pytest -q`: 217 passed.
- Builder compileall, JSON/AST/empty-output checks, ASCII notebook scan, forbidden-control/secret scan, and `git diff --check`: passed.

State128 contract update: the reader-facing header now explicitly states the fixed public runner requirement `1 <= state_len <= 120`, and the focused notebook contract test requires the exact text. The notebook was regenerated; no runtime, CLI-schema, beam-search, or CUDA implementation changed.

Final observed gates after the State128 header update: focused notebook tests `5 passed`; all public CayleyPy tests `184 passed`; full repository Python tests `217 passed`; compileall, generated-notebook JSON load, AST parse for every code cell, empty-output check, ASCII/secret/forbidden-control scan, exact State128 text assertion, and `git diff --check` passed. The notebook was not executed on GPUs and was not pushed.
