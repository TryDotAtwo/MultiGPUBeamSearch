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

## Verified contracts

- anchored release first-mode parsing with strict torchrun-prefix normalization, exact ids/depths/lengths, empty paths, legacy release compatibility, and exact-line debug fallback;
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

Results: runner `41 passed`; focused paths+runner `54 passed`; full public package `97 passed`.

No local CUDA compile/GPU claim: CUDA toolkits are installed, but this Windows environment has no `cl.exe` or configured NCCL build toolchain.