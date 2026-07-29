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

Review-fix update: reader-facing sources now use ASCII-only `--` and `2xT4` text; the checkout URL is a hard-coded official repository rather than a user setting; `SOLVER_COMMIT` is pinned to `6e07844737e782be92ea4dcf49f9c3b14bc4c0ed`; the run cell retains a nonzero CLI return code, then the artifact/status cell displays retained outputs and raises afterwards. The focused test suite now also verifies these properties and ASCII-only notebook emission.

Independent final gate after review fixes:
- `py -m pytest tests/cayleypy_public/test_public_notebook_builder.py -q`: 5 passed.
- `py -m pytest -q`: 217 passed.
- Builder compileall, JSON/AST/empty-output checks, ASCII notebook scan, forbidden-control/secret scan, and `git diff --check`: passed.

State128 contract update: the reader-facing header now explicitly states the fixed public runner requirement `1 <= state_len <= 120`, and the focused notebook contract test requires the exact text. The notebook was regenerated; no runtime, CLI-schema, beam-search, or CUDA implementation changed.

Final observed gates after the State128 header update: focused notebook tests `5 passed`; all public CayleyPy tests `184 passed`; full repository Python tests `217 passed`; compileall, generated-notebook JSON load, AST parse for every code cell, empty-output check, ASCII/secret/forbidden-control scan, exact State128 text assertion, and `git diff --check` passed. The notebook was not executed on GPUs and was not pushed.

## Release hardening follow-up

- Publication remains explicit opt-in. An explicit `RESULTS_INGEST_URL` takes
  precedence; when blank, the CLI reads `CAYLEYPY_RESULTS_INGEST_URL`. No
  endpoint, token, or secret is bundled in the notebook, and an environment
  endpoint does not enable publication by itself.
- A configured or environment endpoint is accepted only as HTTPS with a host
  and without credentials, query, fragment, whitespace, or control bytes.
- Enabled publication fails before export/build/solve when author,
  competition, Kaggle owner/slug, or optional Kaggle username still has a
  `replace-with-*` / `REPLACE_WITH_*` sentinel. Non-publishing local runs retain
  their existing permissive author-placeholder behavior.
- Best-effort runtime behavior is unchanged: HTTP/DNS/service failure remains a
  safe `publish_status.json` outcome after local artifacts are materialized.
- TDD RED: 10 expected config failures, then one focused optional-username
  failure. GREEN: config plus notebook-builder tests `51 passed`; full public
  package `211 passed`; full repository `244 passed`.
- The generated notebook was regenerated from the builder. No GPU execution,
  Kaggle push/publication, external ingest request, beam-search, or CUDA/C++
  change was made in this follow-up.
- Compileall and `git diff --check` passed. Deterministic regeneration retained
  notebook SHA-256
  `c5d79df5d09513d3c4a22f31b40c5560f041070b2ecd41a38bb89980d784d6bf`;
  the changed public builder/helper/notebook secret scan had no hits.

## Public endpoint and redirect hardening

- TDD RED proved that `127.0.0.1`, `::1`, `169.254.169.254`, `localhost`, and
  `.localhost` endpoints passed the former syntactic HTTPS check.
- Config now rejects localhost names and IP literals unless `ipaddress.is_global`
  is true; public DNS names and global IPv4/IPv6 controls remain accepted.
- Publication installs a no-redirect handler. The first 3xx is persisted as a
  bounded safe failure and no `Location` request is issued, including redirects
  toward private HTTPS or plaintext HTTP.
- The hardening was committed locally as
  `bb505484a839d3b78819f86aa28e76b842faab09`; `SOLVER_COMMIT` and its focused
  assertion now pin that exact revision, and the notebook was regenerated.
  Release gate remains OPEN until this commit is reachable from the official
  GitHub repository and the private 2xT4 release smoke passes.
- Observed GREEN after the fix: focused config/results/notebook-builder tests
  `98 passed`; the full public package `221 passed`; `compileall` and
  `git diff --check` passed.
- Generated notebook validation passed JSON load, AST parsing of all code cells,
  exactly six cells, and empty execution outputs. The bounded credential-pattern
  scan was clean. After repinning, current generated notebook SHA-256 is
  `d1f235e64185af33e02cadfd04a47ef4e18dcc8639c22084d4e0497cef7fe20d`.
- No GPU execution, Kaggle push, ingest request, beam-search, or CUDA change occurred.
