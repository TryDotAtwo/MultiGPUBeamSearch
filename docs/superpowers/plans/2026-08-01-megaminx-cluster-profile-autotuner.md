# Megaminx Cluster Profile Autotuner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a standalone, resumable SLURM/torchrun autotuner in every native archive that discovers the maximum stable beam, tunes runtime profiles down to 30,000,000, and emits exact-hardware evidence for an initial 8xA100 40 GB run.

**Architecture:** A login-node wrapper validates immutable session inputs and submits one SLURM job. A pure-Python controller on the compute node derives one hash-budgeted BFS radius, probes the maximum stable beam, runs deterministic successive halving per beam anchor, checkpoints every observation, and emits a fail-closed registry fragment. Existing solver, profile selection, result publication, and Stream 1-5 semantics remain unchanged.

**Tech Stack:** Bash, Python 3 standard library, pytest, SLURM `sbatch`, `torch.distributed.run`, existing native CUDA runner and JSON profile schema.

## Global Constraints

- Default minimum beam is `30000000`; default hard tuning budget is `6h`.
- Default raw BFS hash budget is `256 MiB`; derive the radius once from puzzle `move_count` and stored hash width, with `Hash128=16` bytes for Megaminx.
- Radius is never a successive-halving dimension and is not recomputed per anchor.
- A stable probe requires all ranks, clean CUDA/NCCL, replay and digest success, scratch headroom, per-probe deadline, and at least 10 percent free VRAM on every GPU.
- Final selection requires three measured repetitions across the packaged short, medium, and hard puzzles.
- Hardware identity is exact: GPU family, VRAM class, SM, world size, driver, solver commit, model digest, and release manifest digest.
- Unknown, incomplete, or partial evidence emits `unverified`; only complete final evidence may emit `measured`.
- No GPU, world-size, architecture, model, beam, BFS-radius, PTX, JIT, or runtime fallback.
- Preserve `State128`, `Hash128`, `CandidateMeta`, Stream 3 payload-id, and Stream 4 architecture guardrails from `AGENTS.md`.

---

## File Structure

- `portable/megaminx_cluster/autotune.sh`: public archive-root wrapper.
- `portable/megaminx_cluster/autotune_submit.py`: CLI parsing, session creation, and `sbatch` command.
- `portable/megaminx_cluster/scripts/autotune_job.sh`: compute-node preflight and controller entrypoint.
- `portable/megaminx_cluster/autotune/contracts.py`: immutable dataclasses, checked durations, puzzle/hash contracts, and session identity.
- `portable/megaminx_cluster/autotune/search_space.py`: BFS derivation, beam bounds/anchors, deterministic candidates, and halving.
- `portable/megaminx_cluster/autotune/probe.py`: isolated torchrun invocation, timeout classification, and metric parsing.
- `portable/megaminx_cluster/autotune/evidence.py`: atomic JSON, append-only JSONL, ranking, profile and registry output.
- `portable/megaminx_cluster/autotune/controller.py`: deadline-aware phase orchestration and resume.
- `portable/megaminx_cluster/autotune/calibration.json`: packaged three-puzzle calibration IDs/classes.
- Existing archive workflow/allowlist/README and portable tests are modified to package and verify these files.

---

### Task 1: Checked contracts and formula-derived BFS radius

**Files:**
- Create: `portable/megaminx_cluster/autotune/__init__.py`
- Create: `portable/megaminx_cluster/autotune/contracts.py`
- Create: `portable/megaminx_cluster/autotune/search_space.py`
- Create: `tests/portable/test_megaminx_autotune_search.py`

**Interfaces:**
- Produces: `PuzzleContract(move_count: int, hash_bytes: int, max_bfs_radius: int)`, `BfsBoundary(radius: int, cumulative_states: int, raw_bytes: int, budget_bytes: int)`, `derive_bfs_boundary(contract, budget_bytes)`, `beam_anchors(max_beam, min_beam)`, `refine_max_stable(probe, min_beam)`.

- [ ] **Step 1: Write failing parameterized tests** for Megaminx `(24,16,256*2**20) -> radius=5, states=8308825, raw_bytes=132941200`, a 64-bit hash contract, exact-budget equality, invalid/overflowing fields, schema cap, anchor endpoints, and stable/failing beam refinement.
- [ ] **Step 2: Run** `python -m pytest tests/portable/test_megaminx_autotune_search.py -q` and verify import/implementation failures.
- [ ] **Step 3: Implement checked integer accumulation** that tests the next radius before multiplication, stops at `max_bfs_radius`, rejects a contract whose radius zero cannot fit, and never uses floating point for BFS bytes.
- [ ] **Step 4: Implement deterministic half-up log2 anchors and exponential-plus-binary beam probing** with every attempted beam returned in order for evidence logging.
- [ ] **Step 5: Rerun the focused tests** and require all pass.
- [ ] **Step 6: Commit** with `git add portable/megaminx_cluster/autotune tests/portable/test_megaminx_autotune_search.py && git commit -m "feat: add deterministic autotune search contracts"`.

### Task 2: Deterministic runtime candidate generation and successive halving

**Files:**
- Modify: `portable/megaminx_cluster/autotune/contracts.py`
- Modify: `portable/megaminx_cluster/autotune/search_space.py`
- Create: `tests/portable/test_megaminx_autotune_halving.py`

**Interfaces:**
- Produces: `RuntimeCandidate(config_id: str, runtime: Mapping[str,int])`, `TrialScore(config_id, stable, median_wall_us, peak_vram_mib)`, `candidate_grid(seed_runtime)`, `retain_round(scores, fraction)`, `round_schedule(puzzle_ids)`.

- [ ] **Step 1: Write failing tests** proving only the approved runtime keys vary, config IDs are content-derived and stable, screening uses one warm-up plus one measured run, later rounds expand puzzle coverage, and ties order by stability, median time, peak VRAM, then config ID.
- [ ] **Step 2: Run** `python -m pytest tests/portable/test_megaminx_autotune_halving.py -q` and verify failures.
- [ ] **Step 3: Implement the bounded candidate families** for `b_micro`, Stream 1 concurrency, Stream 3 ring slots, shard count/capacity scale, and Stream 4 batch/trigger/active slots; reject unknown runtime keys.
- [ ] **Step 4: Implement deterministic successive-halving retention** with failed rows ranked after every stable row and a minimum of one survivor.
- [ ] **Step 5: Rerun focused tests** and require all pass.
- [ ] **Step 6: Commit** with `git add portable/megaminx_cluster/autotune tests/portable/test_megaminx_autotune_halving.py && git commit -m "feat: add adaptive profile candidate halving"`.

### Task 3: Login-node CLI and SLURM job contract

**Files:**
- Create: `portable/megaminx_cluster/autotune.sh`
- Create: `portable/megaminx_cluster/autotune_submit.py`
- Create: `portable/megaminx_cluster/scripts/autotune_job.sh`
- Create: `portable/megaminx_cluster/autotune/calibration.json`
- Create: `tests/portable/test_megaminx_autotune_submit.py`
- Create: `tests/portable/test_megaminx_autotune_job_sh.py`

**Interfaces:**
- Produces: `AutotuneSubmitConfig`, `parse_args(argv)`, `build_autotune_sbatch_command(config, archive_root, run_dir, cluster)`, public command `./autotune.sh --gpus ...`.
- Consumes: existing strict `cluster.env` keys and archive-root detection patterns from `submit.py`/`run.sh`.

- [ ] **Step 1: Write failing tests** for defaults, explicit puzzles, unique GPU IDs, `6h` parsing, positive 256 MiB budget, dry-run output, safe SLURM exports, one node/task, requested GPU count, and rejected unknown/unsafe inputs.
- [ ] **Step 2: Run** `python -m pytest tests/portable/test_megaminx_autotune_submit.py tests/portable/test_megaminx_autotune_job_sh.py -q` and verify failures.
- [ ] **Step 3: Implement the wrapper and parser** with defaults `min_beam=30000000`, `time_budget=6h`, `bfs_hash_budget_mib=256`, packaged calibration set, and run directories under `autotune-runs/`.
- [ ] **Step 4: Implement the SLURM script** to verify archive payloads, expose only scheduler-visible GPUs, run the existing hardware preflight, and execute `python3 -m portable.megaminx_cluster.autotune.controller` once; the controller alone launches torchrun probes.
- [ ] **Step 5: Rerun focused tests** and require all pass.
- [ ] **Step 6: Commit** with `git add portable/megaminx_cluster/autotune.sh portable/megaminx_cluster/autotune_submit.py portable/megaminx_cluster/scripts/autotune_job.sh portable/megaminx_cluster/autotune/calibration.json tests/portable/test_megaminx_autotune_submit.py tests/portable/test_megaminx_autotune_job_sh.py && git commit -m "feat: add cluster autotune submission command"`.

### Task 4: Isolated probe runner and strict stability classification

**Files:**
- Create: `portable/megaminx_cluster/autotune/probe.py`
- Create: `tests/portable/test_megaminx_autotune_probe.py`

**Interfaces:**
- Produces: `ProbeRequest`, `ProbeResult`, `build_probe_command(request)`, `run_probe(request, run_command, monotonic)`, `classify_metrics(payload)`.
- Consumes: existing `build_torchrun_command`, native `production_runner`, and per-run metric/log paths.

- [ ] **Step 1: Write failing tests** for one rank per visible GPU, unique rendezvous IDs, complete environment mapping, hard timeout, OOM/CUDA/NCCL/capacity classification, missing metric rejection, replay/digest rejection, disk rejection, and the per-rank 10-percent VRAM margin gate.
- [ ] **Step 2: Run** `python -m pytest tests/portable/test_megaminx_autotune_probe.py -q` and verify failures.
- [ ] **Step 3: Implement subprocess isolation** with a candidate-owned directory, explicit timeout, stdout/stderr files, no shell interpolation, and complete failed `ProbeResult` objects for every exit path.
- [ ] **Step 4: Implement strict metric parsing** requiring requested/effective beam, timings, throughput, host/scratch bytes, all rank peaks/totals, exactness digest, replay result, CUDA/NCCL status, and immutable provenance.
- [ ] **Step 5: Rerun focused tests** and require all pass.
- [ ] **Step 6: Commit** with `git add portable/megaminx_cluster/autotune/probe.py tests/portable/test_megaminx_autotune_probe.py && git commit -m "feat: add strict autotune probe runner"`.

### Task 5: Atomic evidence, resume identity, and registry fragment

**Files:**
- Create: `portable/megaminx_cluster/autotune/evidence.py`
- Create: `tests/portable/test_megaminx_autotune_evidence.py`
- Modify: `portable/megaminx_cluster/profiles/schema.json`

**Interfaces:**
- Produces: `SessionIdentity`, `EvidenceStore.create_or_resume(...)`, `append_trial(result)`, `write_checkpoint(state)`, `write_leaderboard()`, `emit_profile(status)`, `emit_registry_fragment(status)`.

- [ ] **Step 1: Write failing tests** for atomic replacement, append-only JSONL, corrupt-tail rejection, exact identity mismatch on every provenance field, deterministic TSV order, derived BFS fields, and `measured` versus `unverified` emission gates.
- [ ] **Step 2: Run** `python -m pytest tests/portable/test_megaminx_autotune_evidence.py -q` and verify failures.
- [ ] **Step 3: Implement canonical JSON and atomic checkpoints** using same-directory temporary files plus `os.replace`, fsync before replacement, and immutable `session.json`.
- [ ] **Step 4: Implement schema-valid profile output** keyed by exact hardware/world size with evidence IDs for every anchor; refuse `measured` unless all three puzzles have three successful final repetitions.
- [ ] **Step 5: Rerun focused tests and existing profile tests** using `python -m pytest tests/portable/test_megaminx_autotune_evidence.py tests/portable/test_megaminx_cluster_profile.py -q`.
- [ ] **Step 6: Commit** with `git add portable/megaminx_cluster/autotune/evidence.py portable/megaminx_cluster/profiles/schema.json tests/portable/test_megaminx_autotune_evidence.py && git commit -m "feat: persist resumable autotune evidence"`.

### Task 6: Six-hour deadline-aware controller

**Files:**
- Create: `portable/megaminx_cluster/autotune/controller.py`
- Create: `tests/portable/test_megaminx_autotune_controller.py`

**Interfaces:**
- Produces: `BudgetController`, `run_session(config, probe, clock)`, CLI `main(argv)`.
- Consumes: search, probe, and evidence interfaces from Tasks 1-5.

- [ ] **Step 1: Write failing clock-driven tests** for phase ordering, one session BFS derivation, max-beam evidence retention, per-anchor halving, puzzle-order rotation, three final repetitions, reserve calculation, deadline stop, resume from each phase, and partial `unverified` output.
- [ ] **Step 2: Run** `python -m pytest tests/portable/test_megaminx_autotune_controller.py -q` and verify failures.
- [ ] **Step 3: Implement the controller state machine** with explicit phases `max_beam`, `anchors`, `halving`, `finals`, `emit`, checkpointing after every probe.
- [ ] **Step 4: Implement deadline reservation** from observed finalist durations plus bounded setup overhead; stop screening before violating reserve and never start work past the hard deadline.
- [ ] **Step 5: Rerun focused tests** and require all pass.
- [ ] **Step 6: Commit** with `git add portable/megaminx_cluster/autotune/controller.py tests/portable/test_megaminx_autotune_controller.py && git commit -m "feat: orchestrate deadline-aware profile tuning"`.

### Task 7: Archive packaging, documentation, and portable end-to-end gate

**Files:**
- Modify: `tools/megaminx_archive_contract.py`
- Modify: `.github/workflows/megaminx-native-release.yml`
- Modify: `portable/megaminx_cluster/README.md`
- Modify: `tests/portable/test_megaminx_archive_layout.py`
- Modify: `tests/portable/test_megaminx_fixed_allowlist.py`
- Create: `tests/portable/test_megaminx_autotune_e2e.py`
- Modify: `memory/CHANGELOG.md`
- Modify: `memory/PROMPTS.md`

**Interfaces:**
- Produces: each sm75/sm80/sm86/sm89/sm90/sm120 archive contains an executable `autotune.sh`, job script, Python package, calibration JSON, and unchanged signed manifest/hash behavior.

- [ ] **Step 1: Write failing archive and mock-SLURM E2E tests** that build a staged tree, submit a fake job, simulate stable/OOM probes, resume once, and assert all specified outputs and no publication call.
- [ ] **Step 2: Run** `python -m pytest tests/portable/test_megaminx_archive_layout.py tests/portable/test_megaminx_fixed_allowlist.py tests/portable/test_megaminx_autotune_e2e.py -q` and verify failures.
- [ ] **Step 3: Extend the fixed allowlist and release workflow** only for the exact autotune files; preserve manifest payload enumeration and private/secret scans.
- [ ] **Step 4: Document pasteable usage** for 8xA100 40 GB, optional puzzles, budget override, resume, evidence interpretation, and explicit local registry installation.
- [ ] **Step 5: Update project memory files** with the implemented contract and user requirements.
- [ ] **Step 6: Run the complete portable gate** with `python -m pytest tests/portable -q`, `python tools/check_megaminx_native_archive.py --help`, workflow YAML parsing, `git diff --check`, and the existing private/secret scan.
- [ ] **Step 7: Save verification evidence** to `test_results/megaminx_cluster_autotuner_portable_2026-08-01.md` including exact commands and outputs.
- [ ] **Step 8: Commit** with `git add .github/workflows/megaminx-native-release.yml tools/megaminx_archive_contract.py portable/megaminx_cluster tests/portable memory/CHANGELOG.md memory/PROMPTS.md test_results/megaminx_cluster_autotuner_portable_2026-08-01.md && git commit -m "feat: package cluster profile autotuner"`.

### Task 8: Real 8xA100 40 GB qualification

**Files:**
- Create on returned evidence: `test_results/megaminx_autotune_8xa100_40gb_<timestamp>.md`
- Modify after verified run: `portable/megaminx_cluster/profiles/registry.json`
- Modify after verified run: `memory/CHANGELOG.md`

**Interfaces:**
- Consumes: the sm80 native archive and `./autotune.sh --gpus 0,1,2,3,4,5,6,7 --min-beam 30000000 --time-budget 6h`.
- Produces: a runnable 8xA100 profile only if the generated fragment is `measured` and all provenance/gates match.

- [ ] **Step 1: Build and verify the sm80 archive** with its manifest/hash checker and record the archive SHA-256.
- [ ] **Step 2: Run the six-hour cluster command** on one homogeneous 8xA100 40 GB allocation and retain SLURM job ID plus the complete `autotune-runs/` directory.
- [ ] **Step 3: Verify evidence offline**: exact A100/40960 MiB/SM80/world-size-8 identity, BFS radius 5 from 24 moves and 16-byte hashes, all failed probes retained, all finalist repetitions replay/digest clean, and each rank has at least 10 percent VRAM margin.
- [ ] **Step 4: Install the registry fragment only if status is `measured`**; otherwise preserve it as `unverified` and document the exact failed gate without making the profile runnable.
- [ ] **Step 5: Run profile selection tests against representative beams** from 30,000,000 through the discovered maximum and verify no interpolation outside emitted anchors.
- [ ] **Step 6: Record the real-run report and commit** the measured registry/evidence, or commit only the unverified report if qualification fails.
