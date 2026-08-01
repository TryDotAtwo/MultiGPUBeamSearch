# Megaminx Native Cluster Release Design

Date: 2026-08-01
Status: approved design, pending written-spec review

## Objective

Publish a minimal branch and native GitHub Release artifacts that let a user run exactly one Megaminx puzzle through the existing SLURM plus torchrun production path without compiling on the cluster:

```bash
./run.sh --gpus 0,1,2,3 --beam 1000000000 --puzzle 900 --reflect off
```

The launcher preserves local artifacts, validates reported solutions, and sends valid results to the existing Cloudflare results-ingest Worker for GitHub publication.

## Scope and non-goals

The release preserves the current Stream 1-5, memory, history, reflection, schema, and Worker behavior. It does not introduce Docker, cluster compilation, CUTLASS checkout, toolkit installation, model export, PTX, JIT, architecture fallback, mixed-SM ranks, multi-puzzle jobs, Worker task allocation, or GitHub credentials on user clusters.

## Native release matrix

One Linux x86_64 archive is produced for each native target:

| Target | Intended GPUs |
| --- | --- |
| `sm75` | T4 |
| `sm80` | A100, including 40 GB |
| `sm86` | RTX 30-series |
| `sm89` | RTX 40-series and L4 |
| `sm90` | H100 |
| `sm120` | RTX PRO 6000 Blackwell 96 GB class |

Each runner contains cubin for exactly its named SM and no PTX. `cuobjdump` release gates reject missing, additional, or PTX targets. Unknown, mixed, or mismatched devices fail before torchrun. The platform baseline is x86_64 Ubuntu 22.04 or newer. Redistributable runtime libraries are bundled; the NVIDIA driver is not.

## Archive contract

```text
megaminx-smXX/
  run.sh
  cluster.env.example
  bin/production_runner
  lib/
  data/
  weights/
  profiles/
  scripts/job.sh
  scripts/preflight.sh
  scripts/validate_and_publish.py
  MANIFEST.json
  SHA256SUMS
  README.md
```

The manifest records build commit, native SM, minimum driver and glibc, dependency versions, and payload hashes. Source trees, compilers, CUTLASS, build/cache directories, private paths, credentials, and historical logs are forbidden.

## CLI contract

Required options are `--gpus <ids>`, `--beam <positive integer>`, and `--puzzle <nonnegative integer>`. Omitting `--puzzle` prints `missing required --puzzle; specify one Megaminx puzzle id`, returns nonzero, and performs no `sbatch` call. There is no default puzzle.

Optional options are:

- `--reflect off|after|only`, default `off`;
- `--original-solution <path>`, required only for `--reflect only`;
- `--depth <positive integer>` to override the profile default;
- `--publish-only <run-dir>` to retry publication without solving;
- `--dry-run` to print the validated submission without calling `sbatch`.

Unknown values fail closed. Cluster-specific partition, account, QoS, time, CPU, RAM, and scratch settings live in `cluster.env`; ordinary users select only run inputs. Solver tuning stays in versioned per-architecture profiles.

## SLURM and torchrun flow

`run.sh` runs on the login node, verifies hashes and arguments, then submits `scripts/job.sh` with `sbatch`. It prints the parsed job id and run directory. The job requests one node and as many GPUs as listed, maps the allocation to `CUDA_VISIBLE_DEVICES`, and starts one rank per GPU with `python -m torch.distributed.run --nproc-per-node=<count> --no-python` around the prebuilt runner. NCCL and rendezvous ids are unique per SLURM job.

Allocation preflight verifies exact GPU count, one identical SM matching the archive, driver/runtime compatibility, required libraries, VRAM-derived beam capacity, payload hashes, and scratch/history capacity. Failure has no alternate execution path. Rank 0 streams to the main log and other ranks retain separate logs.

## One-puzzle and reflection behavior

Every SLURM job operates on exactly the required puzzle id:

- `off`: solve the original puzzle once.
- `after`: solve original, validate it, build the reflected state, clear only job-owned history, solve reflected, invert the result, and validate it against the original state.
- `only`: validate the supplied original solution, build and solve only the reflected state, invert the result, and validate it against the original.

These modes reuse the existing cluster formulas and semantics. The wrapper never silently switches mode or publishes an unvalidated path.

## Artifacts and publication

Each submission creates an immutable run directory with effective config, SLURM id, release manifest, hardware inventory, per-rank logs, statistics, raw output, CPU replay report, selected original-facing solution, and publication receipt. Requested and aligned effective beam are recorded separately.

The CPU validator independently replays named moves using packaged puzzle data. Only validated results use the existing CayleyPy schema and `POST /v1/results`. The existing Worker continues to validate, queue, authenticate through its GitHub App, and publish append-only GitHub records.

Solve and publication are separate states. Network failure never deletes a solved run. `--publish-only` rechecks hashes and replay, reuses stable submission identity for idempotency, and records the new receipt.

## Failure and cleanup policy

Unsupported or mixed SM, missing `--puzzle`, invalid reflection input, insufficient VRAM/scratch, corrupt content, missing libraries, torchrun/NCCL failure, and failed replay return nonzero with stage and next action.

Cleanup is limited to checked paths inside the run directory. Success may remove job-owned transient history after durable output is written; failure retains history and logs. The launcher never removes an archive, another run, repository checkout, shared dependency, or unchecked environment-derived path.

## Branch and release boundary

Development occurs on `codex/megaminx-native-cluster-release` in an isolated worktree based on `origin/main`. Only portable launch/packaging code, release automation, tests, docs, and project history belong in the branch. Native archives are GitHub Release assets, not tracked blobs. Existing uncommitted main-worktree changes, temporary scripts, caches, builds, private paths, and unrelated experiments are excluded.

## Verification gates

- CLI tests prove missing `--puzzle` makes zero `sbatch` calls.
- Shell syntax, ShellCheck, and mock-SLURM tests cover GPU count, device mapping, job-id parsing, torchrun world size, and reflection modes.
- Negative tests cover unknown/mixed SM, resource shortage, corrupt manifests, and missing libraries.
- `cuobjdump` proves exactly one native SM and no PTX.
- Archive allowlist and secret/private-path scans pass.
- CPU replay fixtures cover `off`, `after`, and `only`.
- Worker mocks cover accepted, duplicate, retryable, and rejected receipts.
- A bounded staging request and subsequent GitHub record are verified.
- Each supported archive receives a real smoke on its target GPU.

An architecture without a real target-GPU smoke may be an explicitly unverified prerelease asset, never a supported release.

## Delivery sequence

1. Implement CLI, job payload, manifest, and deterministic tests.
2. Adapt current one-puzzle/reflection orchestration without data-plane changes.
3. Reuse the existing envelope publisher and Worker contract.
4. Add hermetic per-SM build and packaging automation.
5. Build and inspect all six archives.
6. Smoke T4, A100, RTX 30, RTX 40/L4, H100, and RTX Blackwell as available.
7. Publish passing artifacts to a prerelease, verify Worker-to-GitHub, and then promote verified assets.
