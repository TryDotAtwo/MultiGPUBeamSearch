# Megaminx Native Cluster Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal SLURM/torchrun Megaminx distribution with one native archive per supported SM, beam-width-specific hardware profiles, reflected solve modes, and Cloudflare-to-GitHub result publication.

**Architecture:** A thin Bash entrypoint delegates deterministic validation and profile selection to small Python modules, then submits a fixed SLURM payload. The payload launches the existing prebuilt `production_runner`, validates output independently, and calls the existing result-ingest API. Release tooling builds allowlisted, hashed, single-SM archives and refuses to label untested hardware as supported.

**Tech Stack:** Bash, Python 3.10+ standard library, pytest, SLURM `sbatch`, PyTorch `torch.distributed.run`, CUDA/CUTLASS/NCCL, CMake/Ninja, GitHub Actions/Releases.

## Global Constraints

- Work only on `codex/megaminx-native-cluster-release` in the isolated worktree.
- `--gpus`, `--beam`, and `--puzzle` are mandatory; missing `--puzzle` calls no `sbatch`.
- One SLURM job solves exactly one puzzle; reflection is `off|after|only`.
- Native targets are exactly `75,80,86,89,90,120`; every archive contains one cubin target and no PTX.
- No Docker, compiler, CUDA toolkit, CUTLASS checkout, model export, PTX, JIT, or fallback on user clusters.
- Profiles are exact to GPU family/VRAM, SM, world size, backend/model class, and beam range.
- Requested beam is preserved except documented layout alignment; capacities are derived from the actual request.
- Unknown hardware/profile/configuration fails closed.
- Existing Stream 1-5 and `ARCHITECTURE_NEED.md` contracts remain unchanged.
- Update `memory/CHANGELOG.md`, `memory/PROMPTS.md`, and store verification under `test_results/`.

---

### Task 1: Profile registry and selector

**Files:**
- Create: `portable/megaminx_cluster/profile.py`
- Create: `portable/megaminx_cluster/profiles/schema.json`
- Create: `portable/megaminx_cluster/profiles/registry.json`
- Test: `tests/portable/test_megaminx_cluster_profile.py`

**Interfaces:**
- Produces: `select_profile(registry, HardwareKey, requested_beam, backend, model_class) -> SelectedProfile`.
- Produces: `derive_capacities(SelectedProfile, requested_beam, move_count, output_dim) -> RuntimePlan`.

- [ ] Write failing tests for half-up boundaries, exact hardware/world-size matching, wrong backend, missing range, requested-beam preservation, alignment, and capacity constraints.
- [ ] Run `python -m pytest tests/portable/test_megaminx_cluster_profile.py -q` and confirm failures are missing-module/API failures.
- [ ] Implement immutable dataclasses, strict JSON validation, selection, and capacity derivation using `world_size * shard_count * 1024` alignment.
- [ ] Seed only evidence-backed entries; mark other desired hardware/count tuples `unverified`, which selection rejects.
- [ ] Rerun the focused test and commit `Add native cluster beam profile selector`.

### Task 2: Public CLI and SLURM submission

**Files:**
- Create: `portable/megaminx_cluster/run.sh`
- Create: `portable/megaminx_cluster/submit.py`
- Create: `portable/megaminx_cluster/cluster.env.example`
- Test: `tests/portable/test_megaminx_cluster_submit.py`
- Test: `tests/portable/test_megaminx_cluster_run_sh.py`

**Interfaces:**
- Consumes: Task 1 `select_profile` and `derive_capacities`.
- Produces: `parse_args(argv) -> SubmitConfig`, `build_sbatch_command(config, plan) -> list[str]`, and one-line `submitted_job_id=<id> run_dir=<path>` output.

- [ ] Write failing tests proving each mandatory argument fails before an injected `sbatch` recorder is called, especially exact missing-puzzle text.
- [ ] Add failing tests for GPU list normalization, duplicate ids, reflect contracts, dry-run, job-id parsing, and safe run-directory construction.
- [ ] Run both focused files and confirm expected failures.
- [ ] Implement the minimal Python submitter and Bash wrapper without shell evaluation of user input.
- [ ] Rerun focused tests and commit `Add one-puzzle SLURM launcher`.

### Task 3: Compute preflight and torchrun payload

**Files:**
- Create: `portable/megaminx_cluster/scripts/preflight.py`
- Create: `portable/megaminx_cluster/scripts/job.sh`
- Test: `tests/portable/test_megaminx_cluster_preflight.py`
- Test: `tests/portable/test_megaminx_cluster_job_sh.py`

**Interfaces:**
- Produces: `inspect_gpus(nvidia_smi_csv) -> tuple[GpuInfo, ...]` and `validate_allocation(manifest, plan, gpus, disk) -> PreflightRecord`.
- Consumes environment exported by `submit.py`; launches `bin/production_runner PUZZLE DEPTH REQUESTED_BEAM` under torchrun.

- [ ] Write failing parser/validator tests for exact count, mixed SM, archive mismatch, VRAM shortage, driver floor, hashes, libraries, and scratch budgets.
- [ ] Write a fake `python`/runner harness asserting one rank per GPU, unique rendezvous id, rank logs, and requested beam passed unchanged.
- [ ] Run focused tests and verify RED.
- [ ] Implement preflight JSON and a fail-closed job payload with checked, run-owned cleanup paths.
- [ ] Rerun focused tests and commit `Add native torchrun job payload`.

### Task 4: Original/reflected orchestration and result validation

**Files:**
- Create: `portable/megaminx_cluster/orchestrate.py`
- Create: `portable/megaminx_cluster/validate.py`
- Test: `tests/portable/test_megaminx_cluster_orchestrate.py`
- Test: `tests/portable/test_megaminx_cluster_validate.py`

**Interfaces:**
- Produces: `apply_path`, `invert_path`, `build_reflected_state`, `validate_solution`, and `build_run_steps(reflect_mode, original_solution)`.
- Produces validated JSON records consumed by Task 5.

- [ ] Write failing fixtures for `off`, `after`, and `only`, including invalid original input and inverse roundtrip.
- [ ] Write failing tests for solver-log parsing, CPU replay, original-facing candidate selection, and immutable artifacts.
- [ ] Run focused tests and verify RED.
- [ ] Implement the existing cluster reflection mathematics in focused Python modules and wire `job.sh` to their step plan.
- [ ] Rerun focused tests and commit `Add validated reflection workflows`.

### Task 5: Existing Worker publication client

**Files:**
- Create: `portable/megaminx_cluster/publish.py`
- Create: `portable/megaminx_cluster/scripts/validate_and_publish.py`
- Import/adapt: result envelope rules from `codex/cube4-public-collector:tools/cayleypy_public/results.py`
- Test: `tests/portable/test_megaminx_cluster_publish.py`

**Interfaces:**
- Produces: `build_result_envelope(validated_run) -> dict`, `publish_batch(url, payload) -> PublishReceipt`, and `publish_existing(run_dir)`.

- [ ] Write failing tests against a local HTTP server for 202 accepted, duplicate, retryable 5xx, rejected 4xx, no redirects, and bounded payload.
- [ ] Assert secrets, environment dumps, private paths, weights, and logs never enter the envelope.
- [ ] Run focused tests and verify RED.
- [ ] Implement stdlib HTTPS publication, stable submission/idempotency ids, safe receipt persistence, and `--publish-only`.
- [ ] Rerun focused tests and commit `Add Cloudflare results publication`.

### Task 6: Native archive builder and content gates

**Files:**
- Create: `tools/build_megaminx_native_release.py`
- Create: `tools/check_megaminx_native_archive.py`
- Create: `portable/megaminx_cluster/README.md`
- Test: `tests/portable/test_megaminx_native_release.py`

**Interfaces:**
- Produces deterministic `megaminx-smXX-linux-x86_64.tar.zst`, `MANIFEST.json`, and `SHA256SUMS`.
- Consumes a prebuilt runner, runtime libraries, data, weights, and profiles; never compiles on the cluster.

- [ ] Write failing tests for the exact allowlist, deterministic metadata, forbidden filenames/content, symlink escape, missing assets, and archive naming.
- [ ] Add a fake-`cuobjdump` test proving exactly one SM and no PTX; test all rejection cases.
- [ ] Run focused tests and verify RED.
- [ ] Implement builders/checkers, library dependency closure validation, and public README examples.
- [ ] Rerun focused tests and commit `Add single-SM native release packaging`.

### Task 7: Build automation, profile measurement, and release evidence

**Files:**
- Create: `.github/workflows/megaminx-native-release.yml`
- Create: `tools/measure_megaminx_cluster_profiles.py`
- Create: `test_results/megaminx_native_cluster_release_2026-08-01.md`
- Modify: `memory/CHANGELOG.md`

**Interfaces:**
- Produces per-SM build artifacts as prerelease assets and measured profile records that pass Task 1 schema.

- [ ] Write failing static workflow tests for six targets, pinned CUDA build environments, no PTX flags, artifact checks, and prerelease-only behavior before GPU evidence.
- [ ] Write failing selector-import tests for deterministic sweep winner records with correctness digest and repeated timings.
- [ ] Run tests and verify RED.
- [ ] Implement build matrix, artifact inspection, checksum/secret scans, and profile sweep ingestion.
- [ ] Run the full portable suite, `git diff --check`, forbidden-content scans, and available local build checks.
- [ ] Record exactly which SM/hardware smokes are verified and which remain prerelease-blocked; never claim unavailable hardware was tested.
- [ ] Commit `Add native release automation and verification evidence`.

### Task 8: End-to-end staging and branch handoff

**Files:**
- Modify: `test_results/megaminx_native_cluster_release_2026-08-01.md`
- Modify: `memory/CHANGELOG.md`

**Interfaces:**
- Consumes all prior tasks; produces a reviewable branch and explicit artifact readiness matrix.

- [ ] Run a mock-SLURM end-to-end `off` solve and `after` solve through local validation and fake Worker publication.
- [ ] Build at least the locally possible native artifact and run `cuobjdump`, archive allowlist, hash, and private-field scans.
- [ ] Send one bounded validated fixture to the staging Worker only if it cannot pollute production claims; save the 202/status evidence and verify GitHub appearance when accessible.
- [ ] Run the entire relevant test suite and record command outputs and limitations.
- [ ] Commit final evidence, inspect exact branch diff against `origin/main`, and use `superpowers:finishing-a-development-branch` for handoff.
