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

Original reviewed Task 4 commit: `cfdf7e778c53d3dbf593b61466992cde377a0efb` (`feat: add CayleyPy reflection and path validation`).

## Concerns

`SolutionRecord` includes an explicit `reached_state` field beyond the plan's abbreviated field list because the mandated dedupe key cannot be computed from the listed fields alone. Future runner construction must supply the CPU-replayed terminal state.

## Fix round 1/5

### RED

`python -m pytest tests/cayleypy_public/test_paths.py -q` produced `2 failed, 10 passed`. The failures demonstrated that a caller-owned list mutated `SolutionRecord.reached_state` after construction and that a `None` generator permutation leaked `TypeError` from `validate_original_solution`.

### GREEN

`SolutionRecord.__post_init__` now copies any iterable reached state into a tuple using frozen-dataclass-safe assignment, rejecting a non-iterable state as `ValueError`. `validate_original_solution` now treats only input-contract `TypeError` and `ValueError` as invalid (`False`); the regression confirms unrelated `RuntimeError` still propagates.

### Verification

- `python -m pytest tests/cayleypy_public/test_paths.py -q` -> `12 passed`
- `python -m py_compile tools/cayleypy_public/paths.py tests/cayleypy_public/test_paths.py`
- `git diff --check`
