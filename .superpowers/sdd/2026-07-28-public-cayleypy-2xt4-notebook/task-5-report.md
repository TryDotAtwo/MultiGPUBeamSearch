# Task 5 Report: Existing Runner Orchestration for First and Collect Modes

## Original implementation

Task 5 introduced deterministic two-rank `torch.distributed.run` command construction, first/collect parsing, reflected-search orchestration, `RunArtifacts`, and host-only solve-bucket stop controls. The original reviewed commit was `2f90ed4` (`feat: orchestrate public CayleyPy search modes`).

## Independent review fix round

### RED

The review regressions initially produced 11 failures covering incomplete reflection modes, the real TSV schema, unsafe snapshot sizing, missing invocation artifacts, false-success process parsing, and rank-local collection stops. Further focused RED passes exposed the capacity-sized distributed gather, missing live streaming, decorative profile knobs without manual runtime mode, unbounded history defaults, and leaked 32 GiB static-hybrid scratch arenas. The last focused RED checks were:

- missing manual runtime/history variables and `preflight_history_runtime`: `2 failed, 14 passed`;
- missing per-invocation cleanup contract: the sequential scratch regression failed before launch/cleanup.

### GREEN

Python orchestration now prevalidates all reflection sources before launch; implements `off`, `after_original`, and `only`; creates a standard one-row reflected `test.csv`; validates reflected results in reflected and original orientations; semantically deduplicates final records; and selects the shortest deterministic submission path. The parser consumes the real `solution_path` TSV field and maps `found_depth`, `total_depth`, and touch depth exactly. Nonzero subprocess exits and missing real rank logs are hard failures with prior artifacts retained only for diagnosis.

Every invocation now has a unique rendezvous, torchrun log directory, combined log, both real rank stdout/stderr artifacts, NCCL id path, and history scratch path. Rank-0 tee output is streamed live while the full combined log is flushed incrementally and only a bounded parser tail remains in memory. The manual T4 profile contract is explicit (`manual`, A/B shard buffers, no global spill, Stream5 scale 1.0, 768 MiB headroom). Internal static-hybrid history defaults use two slots, one worker, 28 GiB RAM and 32 GiB disk totals, with C++-matching per-rank capacity and `/tmp` free-space preflight. Scratch cleanup validates the resolved path as a strict descendant of `/tmp/beam_history_public`, runs after spawn/stream/log success or failure, and leaves artifact logs and unrelated paths untouched.

Collect mode sizes the solved snapshot for the full safe local one-depth Stream2 upper bound (`local_beam * move_count`) and fails early on uint32 or necessary T4 memory lower-bound overflow. The existing final scratch buffer is reused for bounded record chunks; host packet vectors are capped at 65,536 records per chunk, and only actual stored counts are gathered. Host stop reasons and true snapshot overflow are synchronized across ranks through the existing stop-flag collective. Explicit stop depth takes precedence over the legacy first-hit window, and status is emitted only for the boundary actually reached.

No CUDA kernel, Stream 1-5 algorithm, device struct, or device-buffer contract was changed.

## Verification

- `python -m py_compile tools/cayleypy_public/runner.py tests/cayleypy_public/test_runner.py tests/cayleypy_public/fixtures/fake_production_runner.py`
- `python -m pytest tests/cayleypy_public/test_runner.py -q` -> `20 passed`
- `python -m pytest tests/cayleypy_public -q` -> `75 passed`
- `git diff --check`

The local Windows checkout has CUDA toolkits but no `cl.exe`/configured NCCL build toolchain, so this fix round does not claim a local `production_runner` compile or GPU run. Coverage for the C++ change is limited to the executable Python chunk-plan mirror and focused C++ source-contract regressions; a real 2xT4 build/run remains the downstream notebook acceptance gate.

## Review notes

The Python solved-snapshot memory check is deliberately a necessary lower bound (snapshot arrays plus current frontier), not a replacement for `production_runner`'s exact `StaticMemoryPlan` and non-static device-budget gate. At the largest profile, unsafe move-count/beam combinations fail closed rather than reducing the requested beam.
