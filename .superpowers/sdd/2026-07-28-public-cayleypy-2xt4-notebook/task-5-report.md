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

## Private real-2xT4 residual acceptance

The downstream acceptance gate is now complete on private Kaggle kernel `trydotatwo/cayleypy-public-task-5-2xt4-gate`, version 2 (`COMPLETE`). Version 1 is excluded because Kaggle title normalization produced a `task-5` slug while the first notebook embedded the pre-normalization slug; the independent provenance validator rejected that mismatch. A notebook/metadata-only correction embedded the actual private slug in version 2.

Version 2 cloned public `origin/main` at exact SHA `6f95bd6bdb32b5f6ef7cca32b96967bce6036503`, overlaid the full reviewed `tools/production_runner.cu` from commit `6830401ed2086921d2563c2bc3c11faf6c5a0741`, and verified its 242,054-byte SHA-256 as `f7d20a2fdec5748052b09804a2b2878cb13f854b8dd29e05db92d6828c223774`. It checked out CUTLASS `afa1772203677c5118fcd82537a9c8fefbcc7008`, built Release for SM75, linked NCCL, and produced binary SHA-256 `c86919b8994ef38f735e6b6159c68198530767bf42a22d17d9bd907c30d6a0ac`.

The gate observed exactly two Tesla T4 GPUs with 15,360 MiB each and used the tracked output-dimension-24 fp16 model. The exact run contract was puzzle 1, beam 65,536, depth limit 8, K1=K2=0, maximum 16 unique paths, solved-result capacity 786,432, and the measured output-move-count p16 profile. First mode returned the real rank-0 release line `solution_length=1 found_depth=1 touch_depth=0 solution=BR` at 0.108281 solver seconds, and independent local replay validated the path.

Collect A and B both completed normally with `capacity_reached`, emitted 16 unique independently valid paths, and produced byte-identical 449-byte TSVs with SHA-256 `74c12063c3f7cd6399546d6dd865d537e966bf8d9b935510174f9d46a92c748e`. Torchrun and both ranks returned zero in all three invocations; all rank stdout/stderr logs, bounded process-tree RSS samples, GPU snapshots, build logs, and source/binary manifests were downloaded. No overflow, OOM, timeout, hang, fatal runtime marker, or secret/private-path scan hit was observed.

The downloaded raw evidence is under `test_results/kaggle_cayleypy_task5_2xt4_gate_v2_2026-07-29/`; the compact verification report is `test_results/kaggle_cayleypy_task5_2xt4_gate_2026-07-29.md`. The independent downloaded-output validator passed, focused builder/validator tests passed 6/6, and the full local suite passed 134/134. Injected rank-0 failure coverage remains source-test-only because no existing safe runtime hook exists. No C++ hook, CUDA kernel, Stream 1-5 algorithm, public publication, Task 6, or Task 7 work was added.
