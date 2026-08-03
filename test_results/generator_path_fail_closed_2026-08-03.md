# Generator-path fail-closed regression - 2026-08-03

- Confirmed root cause: `production_runner` defaulted an unset `BEAM_GENERATOR_PATH` to `FullBeamNice/generators/p900.json`.
- Cube4 p1000 control with explicit `BEAM_GENERATOR_PATH=$DATA/puzzle_info.json`: solved, length 44, found depth 40, touch depth 4, beam 30M.
- Change: `production_runner` now requires a non-empty `BEAM_GENERATOR_PATH` and throws before search if absent.
- TDD RED: focused regression test failed against the fallback.
- GREEN: focused test passed; full `tests/cayleypy_public/test_runner.py` passed (54 tests).
