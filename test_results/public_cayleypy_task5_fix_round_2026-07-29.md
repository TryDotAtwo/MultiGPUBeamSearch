# Public CayleyPy Task 5 independent-review fix evidence - 2026-07-29

Scope: Python orchestration plus minimal host-only collection synchronization/gather controls. CUDA kernels and Stream 1-5 algorithms were not changed.

## TDD evidence

- Initial independent-review regression pass: 11 failures before orchestration hardening.
- Manual runtime/history contract RED: 2 failures, 14 passes.
- History cleanup RED: sequential invocation regression failed before `RunnerInvocation.history_dir` and guarded cleanup were implemented.
- Final focused runner gate: 20 passed.
- Final public package gate: 75 passed.

## Verified contracts

- exact two-rank torchrun, unique rendezvous/log/history paths, real stdout/stderr for both ranks;
- live rank-0 output plus full incremental combined log and bounded parser tail;
- manual measured-profile environment and bounded static-hybrid history preflight/cleanup;
- full reflection source prevalidation and off/after_original/only behavior;
- real TSV `solution_path` schema and exact found/total/touch depth mapping;
- hard failure on nonzero exit or incomplete rank logs with prior diagnostics retained;
- full local-depth snapshot capacity with uint32/T4 lower-bound rejection;
- bounded chunked distributed gather using existing final scratch;
- synchronized capacity/depth/overflow stop decisions and explicit-depth precedence;
- final semantic deduplication and deterministic shortest submission selection.

## Commands

```text
python -m py_compile tools/cayleypy_public/runner.py tests/cayleypy_public/test_runner.py tests/cayleypy_public/fixtures/fake_production_runner.py
python -m pytest tests/cayleypy_public/test_runner.py -q
python -m pytest tests/cayleypy_public -q
git diff --check
```

No local CUDA compile/GPU claim: CUDA toolkits are installed, but this Windows environment has no `cl.exe` or configured NCCL build toolchain.
