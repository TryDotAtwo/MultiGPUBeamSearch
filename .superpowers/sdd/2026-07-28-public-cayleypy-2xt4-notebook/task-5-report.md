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

The downstream gate is complete on private Kaggle kernel `trydotatwo/cayleypy-public-task-5-2xt4-gate`, version 3 (`COMPLETE`). Version 1 remains excluded by the exact-slug gate; version 2 was the first green residual acceptance; version 3 supersedes it after notebook/gate-only hardening. The pushed package was not changed or repushed by the final evidence-sanitization follow-up.

The v3 notebook is 85,360 bytes with SHA-256 `f817a7be1a848918b685e9cb23a0b6c3d5508eeb498c178a1fa7ca08c151fd7a`; metadata SHA-256 is `b9a22006ac26a28127cf0f1f1c162acf41350369ac0159bc0ffc5def97b9bc32`. It cloned public base SHA `6f95bd6bdb32b5f6ef7cca32b96967bce6036503`, overlaid the complete reviewed runner from `6830401ed2086921d2563c2bc3c11faf6c5a0741`, verified its 242,054-byte SHA-256 `f7d20a2fdec5748052b09804a2b2878cb13f854b8dd29e05db92d6828c223774`, checked out CUTLASS `afa1772203677c5118fcd82537a9c8fefbcc7008`, and built Release SM75 with NCCL. The binary SHA-256 is `8dd39f5539fbf88f1d47128834f16e3d66760ef7ce74047758d4d6579b57d53e`.

The gate observed exactly two 15,360 MiB Tesla T4 GPUs and used the tracked output-dimension-24 fp16 model. The exact run contract was puzzle 1, beam 65,536, depth 8, K1=K2=0, maximum 16 unique paths, solved-result capacity 786,432, and the measured output-move-count p16 profile. First mode returned `solution_length=1 found_depth=1 touch_depth=0 solution=BR` at 0.095335 solver seconds, and independent CPU replay validated it.

Collect A and B both completed normally with `capacity_reached`, emitted 16 unique independently valid paths, and produced byte-identical 449-byte TSVs with SHA-256 `74c12063c3f7cd6399546d6dd865d537e966bf8d9b935510174f9d46a92c748e`. Torchrun and both ranks returned zero in all three invocations; all rank stdout/stderr logs, bounded process-tree RSS samples, GPU snapshots, build logs, and source/binary manifests were downloaded. No overflow, OOM, timeout, hang, fatal runtime marker, or secret/private-path scan hit was observed.

### Final private-gate P2 closure

- RED proved the generated cell bypassed its byte cap at EOF and did not terminate/reap a live process group after a capture exception. GREEN routes live and EOF reads through one cap and uses bounded TERM/wait then KILL/wait cleanup without masking the primary error.
- External acceptance no longer trusts notebook-local lifecycle claims. It requires `remote/` to contain exactly `capture_manifest.json`, the v3 push receipt, COMPLETE status, pulled private metadata, and pulled notebook; any extra file or directory fails closed. It hashes and parses the four raw artifacts. The pulled notebook SHA-256 is `4f7562f59ecc66355a9c885c3095de1921735072309a64a047298433d0175753`; after only Kaggle's cell-source list-to-string normalization it is JSON-identical to the pushed notebook. A negative regression proves an extra synthetic author-bearing `list.csv` is rejected without storing a real name.
- The capture window is explicit and modest: push observed `2026-07-29T01:28:41.367231Z`, completion observed `2026-07-29T01:31:29.760778Z`, with only the ordering `push <= completion` asserted. No author identity, list artifact, or remote-run timestamp is retained or claimed.
- The accepted v3 output is under `test_results/kaggle_cayleypy_task5_2xt4_gate_v3_2026-07-29/`; the compact report is `test_results/kaggle_cayleypy_task5_2xt4_gate_2026-07-29.md`. Focused builder/validator tests pass 10/10 and the full suite passes 138/138.

Injected rank-0 failure coverage remains source-test-only because no safe runtime hook exists. No C++ hook, production-runner algorithm, CUDA kernel, Stream 1-5 algorithm, GitHub/public publication, Task 6, or Task 7 work was added.
