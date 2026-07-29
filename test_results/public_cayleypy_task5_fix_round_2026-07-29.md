# Public CayleyPy Task 5 independent-review fix evidence - 2026-07-29

Scope: Python orchestration plus host-only collection control in `production_runner`. CUDA kernels, Stream 1-5 algorithms, device structs, and device-buffer contracts were not changed.

## Round-two TDD evidence

- Release parser/hermetic environment RED: `6 failed, 20 passed`.
- Reflection-source and inverse-closure RED: four intended focused failures.
- Exact source/solver duplicate provenance RED: `2 failed`.
- Bounded cursor collector RED: `4 failed, 31 deselected`.
- Subprocess lifecycle/partial diagnostics RED: `4 failed, 35 deselected`.
- Rank-symmetric processing-error RED: one focused failure.
- Rank-local next-K exchange RED: one focused failure.
- Exact-prefix and finite-seconds parser P2 RED: `6 failed, 2 passed, 41 deselected`.

## Verified contracts

- anchored release first-mode parsing with normalization of exactly `[default0]:`, hostile-prefix rejection, a finite nonnegative decimal seconds grammar with an optional valid exponent, exact ids/depths/lengths, empty paths, legacy release compatibility, and exact-line debug fallback;
- inherited `BEAM_*` and reserved torchrun rank/master/torchelastic variables removed before the explicit invocation environment, while `PATH` and CUDA visibility remain available;
- validated external sources are original-oriented submission/reflection candidates with SHA provenance; independently discovered exact duplicates retain solver provenance;
- inverse closure and all reflection-source paths fail before any subprocess;
- header-only solved reads plus fixed host metadata chunks, bounded rank-local/global next-K sets, strict cursor order including `found_depth`, and cursor rescans across duplicate reconstructed paths;
- only rank-local next-K packets are NCCL-allgathered, with fixed-scratch exchange chunking and the exactness proof captured in source/tests;
- synchronized header overflow, rank-0 accepted count, batch limit, cursor progression, stop reason, and post-reconstruction rank-0 processing errors;
- bounded terminate/wait then kill/wait fallback before scratch deletion, partial combined/available-rank diagnostics, and cleanup notes that do not mask the primary exception;
- no capacity-sized host candidate vector and no stored vector of all accepted collection records.

## Final commands

```text
python -m py_compile tools/cayleypy_public/runner.py tools/cayleypy_public/paths.py tests/cayleypy_public/test_runner.py tests/cayleypy_public/test_paths.py tests/cayleypy_public/fixtures/fake_production_runner.py
python -m pytest tests/cayleypy_public/test_runner.py -q
python -m pytest tests/cayleypy_public/test_paths.py tests/cayleypy_public/test_runner.py -q
python -m pytest tests/cayleypy_public -q
git diff --check
```

Results: runner `49 passed`; focused paths+runner `62 passed`; full public package `105 passed`.

No local CUDA compile/GPU claim: CUDA toolkits are installed, but this Windows environment has no `cl.exe` or configured NCCL build toolchain.
## Private-gate P2 follow-up

Scope remained notebook/gate-only: `production_runner.cu`, CUDA kernels, Stream 1-5 algorithms, device contracts, and public/GitHub publication were untouched.

- RED: the generated gate did not cap bytes read after child EOF and left the process group alive after a capture read exception; the downloaded-output validator also trusted notebook-local claims without independent Kaggle lifecycle evidence.
- GREEN: live and EOF reads share the exact byte cap; every capture/selector/read exception closes capture and attempts bounded TERM/wait then KILL/wait/reap without masking the primary error.
- The external gate now hashes and parses the full Kaggle push/status/list outputs plus pulled private metadata/notebook, requires exact slug/version 3/private/COMPLETE, orders remote `lastRunTime` inside the observed push/completion window, and asserts semantic equality of the pulled notebook.
- Real private v3 completed on two T4s and preserved the accepted CPU-valid `BR` plus byte-identical 16-row collect artifact SHA-256 `74c12063c3f7cd6399546d6dd865d537e966bf8d9b935510174f9d46a92c748e`.
- Final local gates: 9 focused tests and 137 full tests.
