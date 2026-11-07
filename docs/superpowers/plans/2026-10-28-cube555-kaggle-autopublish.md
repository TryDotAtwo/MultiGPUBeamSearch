# Cube555 Kaggle API and publication repair

> **For agentic workers:** Execute inline in the current workspace; do not delegate or start subagents.

**Goal:** Keep the public Cube555 Kaggle notebook aligned with the GitHub-backed CayleyPy API, expose all user-facing search/model controls in its first cell, and prove that a bounded replay-valid Kaggle result travels through Cloudflare to GitHub.

**Architecture:** Preserve `CayleyGraph.beam_search(backend="native")`, the GitHub-pinned source and input bundle, the existing native profile autodetection, and the existing result-publisher client. Make notebook parameters explicit, ensure a requested publication failure is surfaced, and retain the payload/receipts under `cube555_runs`.

**Tech Stack:** Python/Jupyter, CayleyPy, MultiGPUBeamSearch native adapter, Kaggle CLI, GitHub Contents API, Cloudflare results-ingest worker.

**Spec:** User request in the active conversation and `memory/PROMPTS.md`.

## Global Constraints

- No subagents, new rentals, full solve, 140-step test, or repeat of accepted 2.5M/3.5M runs.
- Keep the public notebook readable and its first cell the only user configuration surface.
- Keep GitHub as the pinned source for solver code, model assets, and puzzle data.
- Publish only replay-valid results; retain exact endpoint receipts and verify the matching GitHub record.
- Preserve unrelated dirty workspace changes.

## Review Focus

- Kaggle version metadata must match the version being run.
- `START_STATE_OVERRIDE` must flow consistently into search, replay/save, and publication; restrict it to one selected puzzle.
- A failed or pending Cloudflare receipt must not leave Run All green when publication was requested.
- The native installer commit must be current and immutable, and the notebook must still call CayleyPy's `graph.beam_search` API.
- The bounded proof run must stay below the previously accepted beam/depth workloads.

---

### Task 1: Snapshot current public and local sources

**Files:**
- Read: current Kaggle source v169, GitHub `main` notebook/setup.py, prior run reports and publisher implementation.
- Record: `test_results/kaggle_publication_audit_20261028/`

**Steps:**
- [x] Pull the public Kaggle source and inspect the exact current GitHub files.
- [x] Confirm the mismatch: GitHub local notebook is stale; Kaggle installer URL points to a setup script pinning old solver commit `611d3272`, while current GitHub setup pins `01603caf`.
- [x] Confirm publication code returns `pending_or_failed` without raising and the caller ignores its status.

### Task 2: Repair and validate the notebook contract

**Files:**
- Modify: `kaggle/cube555_github/cube555-native-2xt4-blend.ipynb`
- Validate: `test_results/kaggle_publication_audit_20261028/notebook_contract_test.py`

**Steps:**
- [ ] Add the start-state override and make model, blend, beam, depth, collection, BFS, reflection, timeout, cache, GPU and publication options explicit in the first cell.
- [ ] Pin setup download to current immutable GitHub commit `2ad93dfd46b49e258563aa1a140893c2352db198` (which pins solver source `01603caf`).
- [ ] Pass the configured start state through search, saved replay and publication.
- [ ] Check and record publisher status/receipts; fail the notebook when publication is requested but unconfirmed.
- [ ] Run notebook JSON, code-cell AST and deterministic configuration-contract checks.

### Task 3: Publish synchronized source and notebook

**Files:**
- Update GitHub: `TryDotAtwo/MultiGPUBeamSearch/kaggle/cube555_github/cube555-native-2xt4-blend.ipynb`
- Update Kaggle: `trydotatwo/cube555-native-2xt4-blend`

**Steps:**
- [ ] Update the GitHub notebook using the connected GitHub app and verify the resulting main blob.
- [ ] Run one bounded Kaggle T4×2 result-publication probe with a new config (beam 1,048,576; depth cap 20; puzzle 35) and preserve its outputs.
- [ ] Verify replay, Cloudflare receipt/status and matching GitHub main record; do not repeat 2.5M or 3.5M workloads.
- [ ] Save a final readable public notebook version with normal user defaults without running its full search.

### Task 4: Report and project memory

**Files:**
- Append: `memory/CHANGELOG.md`, `memory/PROMPTS.md`
- Write: `test_results/kaggle_publication_audit_20261028/REPORT.md`

**Steps:**
- [ ] Record notebook version/link, API and first-cell parameters, actual publication route, replay evidence, and remaining constraints.
- [ ] Verify remote GitHub and Kaggle notebook state before claiming completion.
