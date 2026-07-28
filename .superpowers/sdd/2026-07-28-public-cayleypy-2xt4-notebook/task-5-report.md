# Task 5 Report: Existing Runner Orchestration for First and Collect Modes

## Original implementation

Task 5 introduced deterministic two-rank `torch.distributed.run` command construction, first/collect parsing, reflected-search orchestration, `RunArtifacts`, and host-only solve-bucket stop controls. The original reviewed commit was `2f90ed4` (`feat: orchestrate public CayleyPy search modes`).

## Independent review fix round one

The first fix round completed reflection orchestration, real solve-bucket TSV parsing, full local-depth snapshot sizing, hard subprocess/log failures, unique invocation artifacts, live bounded output streaming, manual measured-profile variables, static-hybrid history preflight, guarded scratch cleanup, and synchronized host collection stops. Its standalone commit was `022b5399c8bf81de4bd33693fd927b5137d6ed84`.

## Independent review fix round two

### RED evidence

The second review began from exact commit `022b5399c8bf81de4bd33693fd927b5137d6ed84` and added focused regressions before each implementation slice:

- release first-mode parsing and poisoned inherited environment: `6 failed, 20 passed`;
- external-source records, deterministic reflection-source union, and pre-launch inverse closure: four intended failures;
- solver provenance on an exact external-source duplicate: `2 failed`;
- bounded cursor collection contracts: `4 failed, 31 deselected`;
- live-child stream failure, kill fallback, cleanup masking, and partial artifacts: `4 failed, 35 deselected`;
- rank-symmetric post-reconstruction failure handling: one focused failure;
- rank-local next-K exchange optimization: one focused failure.
- exact-prefix and finite-seconds parser P2: `6 failed, 2 passed, 41 deselected` before the narrow fix.

### GREEN implementation

First-mode parsing now consumes the real anchored release record, including exact puzzle id, `solution_length`, `found_depth`, `touch_depth`, and an empty solution path. It accepts the legacy release record as `found_depth=solution_length, touch_depth=0`, retains exact-line debug `solution_path` compatibility, and normalizes exactly the fixed `--tee=0:3` rank-0 prefix `[default0]:`; hostile lookalike prefixes remain ordinary unrelated text. Release seconds use a nonnegative decimal grammar with an optional valid exponent and must convert to a finite float. Unrelated log text is never searched as a substring. Both first-mode C++ release branches emit found/touch depths. Child environments retain ordinary runtime controls such as `PATH` and `CUDA_VISIBLE_DEVICES`, but remove every inherited `BEAM_*` and reserved torchrun rank/master/torchelastic variable before applying the explicit invocation contract.

Validated external reflection sources are first-class original-oriented solution candidates before any GPU launch and retain their source SHA-256. Source-only solutions can populate the submission, shorter sources beat longer reflected results, and `after_original` uses a deterministic union of external and newly discovered sources. Exact semantic duplicates retain the stronger original/reflected solver provenance. Generator inverse closure is validated globally before the first original or reflected subprocess.

Collect mode reads only the solved header up front. Each rank scans its device metadata in fixed host chunks of at most 65,536 records, retains only its rank-local next-K after the strict cursor `(total_depth, owner_rank, found_depth, parent_idx, route_packed, hash.lo, hash.hi, suffix_id)`, and exchanges only those bounded next-K packets through fixed NCCL scratch. This is exact because any global next-K record omitted from a rank-local next-K would already have K smaller eligible records on that rank. The global batch remains bounded by the synchronized remaining unique target; the cursor advances on the last raw selected record even when reconstructed paths duplicate, so later scans can fill the requested unique count. Overflow and rank-0 accepted counts are synchronized before depth/pass decisions. Rank-0 validation, path conversion, and output exceptions are captured, collectively propagated after each reconstructed record, and only then rethrown, preventing another rank from entering the next reconstruction collective alone.

After `Popen`, any stream or log-capture exception now stops and reaps torchrun before history deletion, using bounded terminate/wait and kill/wait fallback. Partial combined and available rank diagnostics are retained. The original exception remains primary; teardown or scratch-cleanup failures are attached as notes and never replace it. `run_public_search` exposes a diagnostic `RunArtifacts` snapshot whenever an execution was established.

No CUDA kernel, Stream 1-5 algorithm, device struct, or device-buffer contract was changed.

## Verification

- `python -m py_compile tools/cayleypy_public/runner.py tools/cayleypy_public/paths.py tests/cayleypy_public/test_runner.py tests/cayleypy_public/test_paths.py tests/cayleypy_public/fixtures/fake_production_runner.py`
- `python -m pytest tests/cayleypy_public/test_runner.py -q` -> `49 passed`
- `python -m pytest tests/cayleypy_public/test_paths.py tests/cayleypy_public/test_runner.py -q` -> `62 passed`
- `python -m pytest tests/cayleypy_public -q` -> `105 passed`
- `git diff --check`

The local Windows checkout has CUDA toolkits but no `cl.exe` or configured NCCL build toolchain, so this round makes no local `production_runner` compile or GPU-run claim. The C++ host-only changes are covered by executable Python contract mirrors and focused source-contract regressions; a real 2xT4 build/run remains the downstream notebook acceptance gate.