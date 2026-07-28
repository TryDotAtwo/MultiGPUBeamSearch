# Public CayleyPy Task 4 focused gate (2026-07-28)

- RED observed: missing `tools.cayleypy_public.paths` caused expected collection-time `ModuleNotFoundError`.
- GREEN: `python -m pytest tests/cayleypy_public/test_paths.py -q` completed with `9 passed in 0.05s`.
- Static checks are recorded in the Task 4 SDD report.

## Fix round 1/5

- RED: 2 failed, 10 passed for mutable reached-state aliasing and leaked malformed-generator TypeError.
- GREEN: 12 passed in 0.10s; unrelated RuntimeError propagation is covered.
