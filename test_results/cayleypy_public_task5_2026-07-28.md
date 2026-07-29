# Task 5 runner verification — 2026-07-28

- RED: `python -m pytest tests/cayleypy_public/test_runner.py -q` failed with missing `tools.cayleypy_public.runner`.
- GREEN: `python -m pytest tests/cayleypy_public/test_runner.py tests/cayleypy_public/test_paths.py -q` passed: 16 tests.
- Recheck: `python -m pytest tests/cayleypy_public/test_paths.py -q` passed: 12 tests.
- Collection controls are host-only. `BEAM_SOLVE_BUCKET_MAX_SOLUTIONS` caps unique emitted records, while `BEAM_SOLVED_RESULT_CAPACITY` remains per-depth snapshot capacity and overflow still throws.
## 2026-07-29 residual private acceptance follow-up

- Independent-review P2 RED: generated EOF-tail cap and capture-exception cleanup tests both failed; remote evidence drift was accepted.
- GREEN: EOF and live output share one byte cap; capture/read failures close selectors and terminate/reap the process group with bounded TERM/KILL waits; raw remote evidence is hash-bound and independently parsed.
- Private Kaggle v3 is `COMPLETE` on exact 2xT4. List-free remote attestation proves the exact slug, pushed version 3, private metadata, and `push_observed_at <= completion_observed_at`; no author identity or remote-run timestamp is retained or claimed.
- First mode returned CPU-valid `BR`; both collect runs returned 16 unique CPU-valid paths with byte-identical 449-byte TSV SHA-256 `74c12063c3f7cd6399546d6dd865d537e966bf8d9b935510174f9d46a92c748e`.
- Final verification: 9 focused and 137 full tests passed. No production-runner, CUDA-kernel, Stream 1-5 algorithm, GitHub, or public-publication change.
