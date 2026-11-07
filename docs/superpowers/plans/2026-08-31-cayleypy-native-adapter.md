# CayleyPy Native Adapter Implementation Plan

**Goal:** Keep CayleyPy graph/search usage convenient while selecting our unchanged native beam whenever supported.

**Architecture:** Optional Python package `integrations/cayleypy_native`; explicit one-time `enable_native(options)` installs a reversible dispatch for `CayleyGraph.beam_search`. Default `auto` selects native after capability checks or warns with a concrete reason and calls the original method; explicit `native` rejects unsupported calls, `torch` selects the original. Runtime failures after native starts never trigger fallback.

**Tech Stack:** Python >=3.10, CayleyPy/PyTorch, existing CMake/CUDA/NCCL production_runner, subprocess and torchrun. No new CUDA algorithm.

**Spec:** `docs/research/CAYLEYPY_INTEGRATION_RESEARCH_2026-08-31.md` with the user's subsequent approval: wrap our beam, preserve convenient CayleyPy usage, prefer native when possible.

## Global constraints

- Do not edit dirty native source or upstream clone; all implementation is in the new integration package.
- Only exact supported permutation/state/model contracts; smaller state specialization is automatic, not a puzzle-name whitelist.
- No external publication/download/cluster submission. Local native build needs explicitly configured source and CUTLASS paths.
- Source fingerprint must account for working-tree source, not just Git SHA. Shape-specific builds explicitly pass logical bytes and move count, never reuse stale CMake dimensions.
- Model export only recognized existing native schemas; arbitrary upstream models fall back before launch. Bind exported weights to ordered graph fingerprint.
- Preserve native scoring/dedup/global beam. Reconstruct and replay every successful path against the original graph, even if caller did not request path.
- Keep CUDA visibility/rank/input/history environment private to the child process; disable inherited solve/repair/publication modes.
- Verification on this Windows host is CPU/API/process-contract evidence; do not claim CUDA solve without a real run.

## Task 1: Graph/model boundary

Create `cayleypy_native/contracts.py`, `models.py`, corresponding tests. Define `GraphContract` from CayleyGraph + start with exact gather semantics; validate shape/integer/multiset/ranges and synthetic native move IDs. `prepare_model(predictor, contract, options, run_dir)` accepts a graph-bound `NativeModel` or a recognized loaded torch model from Predictor and reuses source MLP exporter on a safely written tensor-only checkpoint. Unsupported model raises NativeUnavailable; corrupt artifacts/errors raise NativeBackendError.

- [x] Test arbitrary valid move names/order, colored states, invalid/unsupported matrices/ranges, replay.
- [x] Test graph fingerprint mismatch, unsupported upstream MLP, known native artifact dimensions and files; never deserialize untrusted arbitrary Python.
- [x] Implement boundary and run tests.

## Task 2: Native process/build boundary

Create `backend.py`, `build.py`, corresponding tests. `runtime_devices(graph, options)` validates Linux/CUDA, visible selected devices, nesting and build prerequisites without starting jobs. `run_native(contract, model, options, beam_width, max_steps, run_dir, devices)` writes exact puzzle/start inputs, builds or validates binary, runs rank process(es), parses terminal result, returns `NativeOutcome` after replay. Build path derives from source content, shape, architecture, backend/toolchain and artifact/build configuration; use existing binary only with matching build metadata. Log all commands and preserve paths.

- [x] Test shell-free invocation, env cleanup, input order, missing terminal record, crash, timeout and invalid replay.
- [x] Test build-key changes when shape/source/model backend changes, explicit shape flags, no stale cache reuse.
- [x] Implement subprocess runner and validate with an explicitly labelled fake runner fixture, without claiming CUDA validation.

## Task 3: Dispatch, packaging, docs

Create `options.py`, `errors.py`, `dispatch.py`, `__init__.py`, `pyproject.toml`, README and example. Dispatch methods expose `auto/native/torch`, fallback warning, `NativeModel` and telemetry, restore original method via `disable_native()`. Preserve original kwargs on fallback; unsupported advanced/history/custom target/MITM options do not silently disappear. No import-time patch, CUDA work, installation or mutation.

- [x] Test reversible/idempotent installation, auto fallback exactly once, strict native error, explicit torch, no fallback after backend error, upstream result compatibility and exact replay.
- [x] Run existing bounded CayleyPy tests with wrapper disabled and enabled on CPU, package installation in an isolated target directory, and full integration test suite.
- [x] Review native launch correctness independently, update project journals and record exact unverified GPU work.

Implementation proceeds in this task using bounded parallel agents; no additional approval is required for the already approved scope. No commits or changes to unrelated files are planned.

The implementation and CPU/package acceptance are complete. Subsequent user-
authorized private Kaggle v4 validation passed181 Linux CPU tests,12 synthetic
Torch/native scenarios, two explicit prepared-model calls and ten real Tetraminx
runs. Sixteen calls actually launched both T4 ranks; successful paths replayed.
Native/CayleyPy source trees stayed clean. Evidence and full limitations are in
test_results/cayleypy_native_2xt4_2026-08-31/README.md. This verifies two-GPU
integration, not a one-GPU scaling baseline, deep-search throughput or trained
Torch speedup. The unrelated shared GPU queue was not reconfigured.
