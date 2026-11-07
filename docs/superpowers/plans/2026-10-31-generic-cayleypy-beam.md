# Generic CayleyPy Beam Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the public notebook and package's `multigpubeamsearch.cube555` helper dependency while preserving the native CayleyPy beam-search workflow.

**Architecture:** Keep `graph.beam_search` and the native backend as-is. Promote path handling, replay validation, and result serialization into the general library API; make puzzle identity and publisher metadata explicit inputs. The notebook defines a graph, initial states, and predictor, then uses the same generic search controls.

**Tech Stack:** Python, CayleyPy, `multigpubeamsearch`, Jupyter/Kaggle.

**Spec:** `docs/superpowers/specs/2026-10-31-generic-cayleypy-beam-design.md`

## Global Constraints

- Keep `multigpubeamsearch` as the native execution package.
- Keep CayleyPy's public `graph.beam_search(...)` entry point.
- Preserve all current search, collect, BFS, reflection, profile, and result options.
- Native models must use a registered exporter/backend.

## Review Focus

- A path is accepted only when its moves resolve against this graph and replay to its center state.
- A different graph cannot reuse a graph-bound model artifact silently.
- Reflection paths are validated in the original graph frame.
- Publishing without the caller's puzzle/model identity must fail clearly or remain disabled.
- The public notebook must still run its Cube555 default example.

---

### Task 1: Generic path and result helpers

**Files:**
- Modify: `integrations/cayleypy_native/multigpubeamsearch/results.py`
- Modify: `integrations/cayleypy_native/multigpubeamsearch/__init__.py`
- Delete: `integrations/cayleypy_native/multigpubeamsearch/cube555.py`
- Delete: `integrations/cayleypy_native/tests/test_cube555_example.py`

- [x] Expose generic move-name resolution, replay validation, and atomic result saving using `GraphContract`.
- [x] Remove the puzzle-specific helper and replace its user-facing result path with generic helpers.
- [x] Run focused graph/result API checks.

### Task 2: Generic publication metadata

**Files:**
- Modify: `integrations/cayleypy_native/multigpubeamsearch/publication.py`
- Modify: `integrations/cayleypy_native/multigpubeamsearch/__init__.py`

- [x] Accept explicit graph, competition, puzzle-type, and model metadata; do not assume a Cube555 catalog identity.
- [x] Keep the exact Cube555 model metadata only when the native members are the Cube555 Transformer and MLP families.
- [x] Validate generic payload construction with a fake client and catalog-shaped metadata.

### Task 3: Public notebook problem boundary

**Files:**
- Modify: `kaggle/cube555_github/cube555-native-2xt4-blend.ipynb`
- Modify: `kaggle/cube555_github/README.md`
- Modify: `memory/CHANGELOG.md`
- Modify: `memory/PROMPTS.md`

- [x] Remove the notebook import/calls to `multigpubeamsearch.cube555`.
- [x] Accept a CayleyPy `GRAPH`, direct `PREDICTOR`, and direct starting state(s); retain the Cube555 model builder only as the notebook default.
- [x] Route reflection, replay, result saving, and publication through generic beam APIs.
- [x] Preserve first-cell search controls and state native model and catalog limits.
- [x] Publish the exact-source notebook and package changes; verify the public revision without starting a 140-step GPU search.

**Verification:** Implementation release on GitHub main is `5f00f003d87374ee704f3629aef770ae8f20e420`; verification log follow-up is main `0a18735939dca0c1867e76931acfabeb0be16c56`. Kaggle notebook v180 was saved with `--no-run`, pulled back, and all seven cell sources matched the published notebook. Version-specific Kaggle status was `COMPLETE`. Ten focused local tests passed. No GPU search was started for this change.
