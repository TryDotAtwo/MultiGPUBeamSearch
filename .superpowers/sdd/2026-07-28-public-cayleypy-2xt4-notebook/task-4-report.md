# Task 4 Report: Path Replay, Reflection, and Deduplication

## RED

`python -m pytest tests/cayleypy_public/test_paths.py -q` failed during collection with `ModuleNotFoundError: No module named 'tools.cayleypy_public.paths'`. The test file specified dot-separated replay, algebraic inverse lookup without `-` names, reflection inversion, invalid-token rejection, frozen records, and semantic deduplication.

## GREEN

Implemented `tools/cayleypy_public/paths.py` with strict permutation validation, `.`-separated paths, inverse-permutation lookup requiring exactly one named inverse, reflected states derived from the central state, and CPU validation. `SolutionRecord` is frozen and includes the reached state required by the specified deduplication key. Deduplication hashes the original-oriented path and reached state, preserves first-seen key order, and selects the lowest deterministic provenance tuple for duplicates.

## Verification

- `python -m pytest tests/cayleypy_public/test_paths.py -q` -> `9 passed`
- `python -m py_compile tools/cayleypy_public/paths.py tests/cayleypy_public/test_paths.py`
- `git diff --check`

## Self-review

The implementation never infers an inverse from a name; duplicate or absent inverse permutations fail before inversion. Empty internal path fields, unknown moves, malformed permutations, and invalid state lengths fail closed. No CUDA/C++ or unrelated solver changes were made.

## Commit

7bd1a7c (`feat: add CayleyPy reflection and path validation`).

## Concerns

`SolutionRecord` includes an explicit `reached_state` field beyond the plan's abbreviated field list because the mandated dedupe key cannot be computed from the listed fields alone. Future runner construction must supply the CPU-replayed terminal state.