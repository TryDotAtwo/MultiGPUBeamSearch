# Repository Branch Structure Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mixed legacy branch graph with a protected `main`, one tested long-lived `stream1` integration branch, and short-lived task branches without losing local work, cluster launchers, Kaggle notebooks, publishing services, tags, or result evidence.

**Architecture:** The migration is append-only until every preservation and verification gate passes. Existing local and remote refs remain immutable under `legacy/2026-08-08/*`; clean canonical branches are reconstructed from an explicit base and receive only subsystem-owned commits. `stream1` is the sole long-lived architecture integration branch; MLP, Transformer, common contracts, and future backend tasks use short branches from `stream1` and merge back only after cross-architecture tests pass.

**Tech Stack:** Git worktrees and bundles, GitHub CLI/API, CMake/CTest, Python/pytest, Bash syntax validation, JSON/Jupyter notebook validation, GitHub branch protection.

## Global Constraints

- Do not force-push or rewrite any existing published commit, branch, or tag.
- Do not remove a local branch, remote branch, worktree, patch, notebook, cluster launcher, or test artifact until its object id or content hash appears in the preservation manifest.
- Treat `main`, Stream 1, cluster/HPC, Kaggle notebooks, and results publishing as separate verification domains.
- Keep `State128`, `Hash128`, `CandidateMeta`, Stream 3 payload, and Stream 4 behavior exactly as specified by `AGENTS.md` and `ARCHITECTURE_NEED.md`.
- Every Codex-authored migration commit must contain `Co-authored-by: OpenAI Codex (GPT-5.6 Sol) <codex@openai.com>`.
- Record requirement prompts in `memory/PROMPTS.md`, meaningful changes in `memory/CHANGELOG.md`, and executed verification output under `test_results/`.
- The existing dirty checkout at `D:/100XH100` is evidence, not a staging area; do not clean, reset, stash, or commit it during migration.

---

### Task 1: Freeze a lossless repository inventory

**Files:**
- Create: `test_results/branch_migration_inventory_2026-08-08.txt`
- Create: `test_results/branch_migration_worktrees_2026-08-08.txt`
- Create: `test_results/branch_migration_github_2026-08-08.json`
- Create: `test_results/branch_migration_sha256_2026-08-08.txt`

**Interfaces:**
- Consumes: local Git object database, worktree registry, `TryDotAtwo/MultiGPUBeamSearch` GitHub repository.
- Produces: immutable ref/object/content inventory used by every deletion gate.

- [x] **Step 1: Capture every local and remote ref without updating it**

Run from `D:/100XH100`:

```powershell
git show-ref | Sort-Object | Set-Content test_results/branch_migration_inventory_2026-08-08.txt
git ls-remote --heads --tags origin | Sort-Object | Add-Content test_results/branch_migration_inventory_2026-08-08.txt
git fsck --full --no-reflogs --unreachable | Add-Content test_results/branch_migration_inventory_2026-08-08.txt
```

Expected: exit code `0`; the file contains `refs/heads/main`, `refs/remotes/origin/main`, all feature refs, and tags.

- [x] **Step 2: Capture worktree paths, branches, HEADs, and dirty file lists**

```powershell
git worktree list --porcelain | Set-Content test_results/branch_migration_worktrees_2026-08-08.txt
$paths = git worktree list --porcelain | Select-String '^worktree ' | ForEach-Object { $_.Line.Substring(9) }
foreach ($path in $paths) {
  "`n[$path]" | Add-Content test_results/branch_migration_worktrees_2026-08-08.txt
  if (Test-Path -LiteralPath $path) {
    git -C $path status --short --untracked-files=all | Add-Content test_results/branch_migration_worktrees_2026-08-08.txt
  } else {
    'MISSING' | Add-Content test_results/branch_migration_worktrees_2026-08-08.txt
  }
}
```

Expected: the dirty root checkout and every linked, detached, missing, and prunable worktree are represented.

- [x] **Step 3: Capture GitHub branches and pull requests through the authenticated API**

```powershell
$repo = 'TryDotAtwo/MultiGPUBeamSearch'
$branches = gh api --paginate "repos/$repo/branches"
$pulls = gh pr list --repo $repo --state all --limit 200 --json number,title,state,isDraft,headRefName,baseRefName,mergedAt,closedAt,updatedAt,url
@{ repository=$repo; captured_at=(Get-Date).ToUniversalTime().ToString('o'); branches=($branches | ConvertFrom-Json); pulls=($pulls | ConvertFrom-Json) } |
  ConvertTo-Json -Depth 20 |
  Set-Content test_results/branch_migration_github_2026-08-08.json
```

Expected: valid JSON containing remote `main`, all feature branches, and PRs `#1` and `#2`.

- [x] **Step 4: Hash every dirty or untracked file without modifying it**

Use `git status --porcelain=v1 -z` in each existing worktree, resolve paths relative to that worktree, and run `Get-FileHash -Algorithm SHA256` for regular files. Store `worktree`, `status`, relative path, byte length, and SHA-256 in `test_results/branch_migration_sha256_2026-08-08.txt`.

Expected: every material file listed in the worktree inventory has either a SHA-256 row or an explicit `directory` row.

- [x] **Step 5: Verify the inventory is internally complete**

```powershell
git show-ref --verify refs/heads/main
git cat-file -e b5fcf6b0ee3247a1f189e6da79a1641d2304bd1c^{commit}
git cat-file -e 6f95bd6bdb32b5f6ef7cca32b96967bce6036503^{commit}
git cat-file -e f769e707dcda2b962bbaf99c62b40a1fd88f8108^{commit}
```

Expected: all commands exit `0`.

- [x] **Step 6: Commit only the inventory artifacts**

Commit title: `chore: inventory repository before branch migration`

Commit trailer: `Co-authored-by: OpenAI Codex (GPT-5.6 Sol) <codex@openai.com>`

---

### Task 2: Create recoverable legacy refs and a Git bundle

**Files:**
- Create outside the repository: `D:/100XH100-backups/MultiGPUBeamSearch-before-branch-migration-2026-08-08.bundle`
- Create: `test_results/branch_migration_bundle_verify_2026-08-08.txt`

**Interfaces:**
- Consumes: Task 1 ref inventory.
- Produces: offline recovery bundle and `legacy/2026-08-08/*` refs.

- [x] **Step 1: Create legacy refs for every local branch tip**

For each `refs/heads/<name>` in the Task 1 inventory, create `refs/heads/legacy/2026-08-08/<name>` with `git update-ref` and the exact recorded object id. Do not move the original ref.

- [x] **Step 2: Create legacy refs for remote-only tips**

Create explicit legacy refs for remote `stream1-real-state-rank-parity` at `f769e707dcda2b962bbaf99c62b40a1fd88f8108` and every other remote SHA absent from local heads.

- [x] **Step 3: Create a full Git bundle**

```powershell
New-Item -ItemType Directory -Force -Path 'D:/100XH100-backups'
git bundle create 'D:/100XH100-backups/MultiGPUBeamSearch-before-branch-migration-2026-08-08.bundle' --all
git bundle verify 'D:/100XH100-backups/MultiGPUBeamSearch-before-branch-migration-2026-08-08.bundle' |
  Tee-Object test_results/branch_migration_bundle_verify_2026-08-08.txt
```

Expected: bundle verification reports a complete history and exits `0`.

- [x] **Step 4: Prove three representative tips are recoverable from the bundle**

Clone the bundle into a temporary directory and verify the exact object ids for `main`, `codex/stream1-piece-transformer`, and `codex/megaminx-native-cluster-release`. Remove only that newly created temporary clone after verification.

- [x] **Step 5: Commit the bundle verification note**

Commit title: `chore: verify pre-migration Git recovery bundle`

---

### Task 3: Establish development attribution and branch policy

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Create: `CONTRIBUTORS.md`
- Create: `.github/pull_request_template.md`
- Modify: `memory/PROMPTS.md`
- Modify: `memory/CHANGELOG.md`

**Interfaces:**
- Consumes: user requirement that GPT-5.6 Sol Codex performs implementation.
- Produces: repository-visible attribution and enforceable contribution convention.

- [x] **Step 1: Add the commit attribution contract to `AGENTS.md`**

Add an `AI Development Attribution` section requiring the exact trailer:

```text
Co-authored-by: OpenAI Codex (GPT-5.6 Sol) <codex@openai.com>
```

for commits substantially implemented by Codex.

- [x] **Step 2: Add a concise development statement to `README.md`**

State that Ivan Litvak directs, reviews, and accepts the project, while OpenAI Codex running GPT-5.6 Sol performs the primary architecture, implementation, testing, profiling support, and documentation work.

- [x] **Step 3: Create `CONTRIBUTORS.md` with role boundaries**

List human project ownership separately from Codex implementation responsibilities. State that historic commits are documented here rather than rewritten.

- [x] **Step 4: Add PR attribution and subsystem gates**

The PR template must require: source branch, target branch, subsystem (`stream1`, `cluster`, `kaggle`, `publishing`, `docs`), Codex model, human direction, exact verification commands, hardware used, and unchanged-domain checks.

- [x] **Step 5: Record the requirements and change**

Append the current branch-cleanup, single Stream 1 integration branch, no-regression, and GPT-5.6 Sol attribution requirements to `memory/PROMPTS.md`; append the policy files and migration plan to `memory/CHANGELOG.md`.

- [x] **Step 6: Verify attribution text**

```powershell
rg -n "GPT-5\.6 Sol|codex@openai\.com|Stream 1|cluster|Kaggle" AGENTS.md README.md CONTRIBUTORS.md .github/pull_request_template.md memory/PROMPTS.md memory/CHANGELOG.md
```

Expected: every policy surface contains its intended statement.

- [ ] **Step 7: Commit the policy files**

Commit title: `docs: define branch and Codex attribution policy`

---

### Task 4: Reconstruct the canonical Stream 1 integration branch

**Files:**
- Create branch: `stream1`
- Create temporary branches: `migration/stream1-piece-transformer`, `migration/stream1-sequence-alignment`, `migration/stream1-real-state-parity`
- Create: `test_results/stream1_branch_migration_2026-08-08.md`

**Interfaces:**
- Consumes: `codex/stream1-piece-transformer`, remote `codex/stream1-real-state-rank-parity`, `codex/stream1-generic-seq-align-impl`, `codex/stream1-transformer-accumulator`, and `codex/kaggle-t4-mlp-autoprofiles`.
- Produces: one buildable long-lived `stream1` branch with MLP, Transformer, and common architecture support.

- [ ] **Step 1: Classify commits by subsystem before replay**

Generate `git log --reverse --format='%H%x09%s'` and `git diff-tree --no-commit-id --name-only -r` tables for each source range. Mark a commit Stream 1-owned only when all material files belong to CUDA/model export/runtime/config/tests/docs for Stream 1. Mark cluster, Kaggle notebook, publishing, result data, and unrelated documentation commits for their own migrations.

- [ ] **Step 2: Compare the local and GitHub parity tips**

```powershell
git fetch origin refs/heads/codex/stream1-real-state-rank-parity:refs/remotes/origin/codex/stream1-real-state-rank-parity
git log --oneline d319f6db0540b6ccef0d41258407815cfe163cce..f769e707dcda2b962bbaf99c62b40a1fd88f8108
git diff --stat d319f6db0540b6ccef0d41258407815cfe163cce..f769e707dcda2b962bbaf99c62b40a1fd88f8108
```

Expected: the exact remote-only commits and files are recorded before choosing the parity tip.

- [ ] **Step 3: Create `stream1` from the migration base**

Create `stream1` at the branch-policy commit from Task 3. Replay only classified Stream 1 commits in their original dependency order, preserving original author data and adding the Codex co-author trailer only to new migration/integration commits, not by rewriting historical commits.

- [ ] **Step 4: Keep architecture boundaries explicit**

The resulting tree must retain MLP, Transformer, common input/checkpoint contracts, backend selection, export tooling, exactness tests, and architecture-specific profiling. Do not move cluster launchers, public notebooks, result publishing, or generated evidence onto `stream1` merely because their source branch contains them.

- [ ] **Step 5: Validate patch equivalence for each accepted Stream 1 series**

Use `git range-diff` between each classified source series and its replayed series. For squash or conflict-resolution commits, use `git diff --no-renames --binary <source-tree> <reconstructed-tree>` limited to the classified paths and require an empty diff.

- [ ] **Step 6: Run Stream 1 host and CUDA gates**

Configure a fresh build directory and run the complete CTest suite available on the current machine, including `stream1_cuda_tests`. Run Python export/config/checkpoint tests discovered under `tests/`. If CUDA hardware or the required toolchain is unavailable, record the exact missing executable/device and do not mark the hardware gate passed.

- [ ] **Step 7: Run cross-architecture gates**

Run exact score-dump comparisons for MLP and Transformer reference fixtures, production-runner configuration parsing, checkpoint metadata validation, and CUDA architecture/static-memory tests. A change accepted for one architecture must not change the other architecture's expected digest.

- [ ] **Step 8: Record Stream 1 evidence and commit the integration state**

Write commit mappings, range-diff results, test commands, exit codes, hardware identity, and score digests to `test_results/stream1_branch_migration_2026-08-08.md`.

Commit title: `chore: establish canonical Stream 1 integration branch`

---

### Task 5: Preserve and validate cluster/HPC work independently

**Files:**
- Create branch: `cluster/production`
- Create: `test_results/cluster_branch_migration_2026-08-08.md`

**Interfaces:**
- Consumes: `codex/megaminx-native-cluster-release`, root dirty HPC files, detached `megaminx-native-cluster-fix`, and cluster-related commits embedded in other legacy branches.
- Produces: a cluster branch whose launcher semantics and binary contracts match preserved sources.

- [ ] **Step 1: Reconstruct cluster-owned commits from the Task 1 inventory**

Include `hpc/**`, cluster-facing production-runner configuration required by those scripts, and their tests/docs. Exclude Stream 1 model kernels, Kaggle notebooks, publishing services, and generated result archives unless the cluster contract directly consumes them.

- [ ] **Step 2: Preserve dirty root HPC files as a separate patch artifact**

Create a binary-safe `git diff --binary` for tracked modifications and a tar/zip archive for untracked regular files listed in the SHA manifest. Do not apply them automatically; compare them path-by-path against `cluster/production` and record whether each is already present, needs replay, or is generated scratch.

- [ ] **Step 3: Verify detached cluster work is contained**

Prove commit `847fe2a46fcf2135155ca9c368f5813f73ab9f75` is an ancestor of the reconstructed branch with `git merge-base --is-ancestor`.

- [ ] **Step 4: Validate every shell launcher**

Run `bash -n` through WSL or the project Docker image for every tracked `*.sh` under `hpc/`. Record the exact file list and require zero syntax failures.

- [ ] **Step 5: Validate configuration and dry-run contracts**

Run existing launcher/configuration tests and safe dry-run modes without submitting SLURM jobs or launching production GPU workloads. Verify `WORLD_SIZE`, GPU topology, prebuilt runner reuse, resume paths, result paths, and Stream 1 backend selection remain explicit.

- [ ] **Step 6: Record evidence and commit**

Commit title: `chore: establish canonical cluster production branch`

---

### Task 6: Preserve and validate Kaggle notebooks

**Files:**
- Create branch: `kaggle/notebooks`
- Create: `test_results/kaggle_branch_migration_2026-08-08.md`

**Interfaces:**
- Consumes: `codex/public-cayleypy-notebook`, `codex/cube4-public-collector`, `codex/profile-capacity-safety`, `codex/cube4-fp16-parity`, and Kaggle files embedded in Stream 1 branches.
- Produces: parseable, reproducible notebooks and builders pinned to preserved solver commits.

- [ ] **Step 1: Classify notebooks, builders, configs, and acceptance fixtures**

Map every tracked `*.ipynb`, notebook builder, Kaggle metadata file, public runner module, and notebook-specific test to its source commit and SHA-256.

- [ ] **Step 2: Reconstruct `kaggle/notebooks` without publishing-service history**

Replay notebook-owned commits from the Task 3 base. Keep result-ingest service files on the publishing branch and Stream 1 kernels on `stream1`; notebook pins may reference immutable solver commits or release tags.

- [ ] **Step 3: Parse every notebook and metadata file**

Run Python `json.load` for all `*.ipynb` and Kaggle JSON metadata. Require zero parse failures and validate required kernel metadata fields.

- [ ] **Step 4: Regenerate generated notebooks and compare them**

Run each tracked notebook builder into a temporary directory. Compare generated notebooks with canonical tracked notebooks after stripping only explicitly nondeterministic notebook metadata. Any code-cell difference is a failed gate.

- [ ] **Step 5: Run notebook/public-runner tests**

Run the repository's CayleyPy public config, builder, result, schema, capacity, and FP16 parity tests. Do not claim Kaggle GPU compatibility from local parsing alone; preserve existing Kaggle evidence and mark live reruns separately.

- [ ] **Step 6: Record evidence and commit**

Commit title: `chore: establish canonical Kaggle notebook branch`

---

### Task 7: Preserve and validate result publishing

**Files:**
- Create branch: `publishing/results`
- Create: `test_results/publishing_branch_migration_2026-08-08.md`

**Interfaces:**
- Consumes: `codex/cayleypy-results-ingest`, collector/publisher commits, GitHub workflows, schemas, and publishing tests.
- Produces: one publishing branch with the ingest service and deterministic result publisher, independent of solver development.

- [ ] **Step 1: Separate publishing-owned history**

Classify `services/cayleypy-results-ingest/**`, publisher plugin/client/archive builder, schemas, publishing workflows, migrations, tests, and runbooks. Exclude CUDA, cluster scheduling, and notebook implementation except shared schemas consumed through a documented contract.

- [ ] **Step 2: Reconstruct `publishing/results` from the Task 3 base**

Replay only publishing-owned commits. Preserve migration order and GitHub/Cloudflare deployment configuration exactly.

- [ ] **Step 3: Run service and publisher gates**

Install dependencies from the committed lockfile in the isolated worktree; run service tests, type checking, schema tests, archive determinism tests, replay/idempotency tests, and publisher recovery tests. Do not deploy or mutate live Cloudflare/GitHub state during this gate.

- [ ] **Step 4: Compare PR #1 tree content**

For every publishing-owned file in PR #1, compare blob hashes or binary diffs with `publishing/results`. Record intentional exclusions with their destination branch.

- [ ] **Step 5: Record evidence and commit**

Commit title: `chore: establish canonical results publishing branch`

---

### Task 8: Integrate verified branches and replace oversized PRs

**Files:**
- Modify: GitHub branch refs and PR metadata only after Tasks 1–7 pass.
- Create: `test_results/branch_migration_integration_2026-08-08.md`

**Interfaces:**
- Consumes: verified `stream1`, `cluster/production`, `kaggle/notebooks`, and `publishing/results` branches.
- Produces: reviewable GitHub branches and PRs with preserved legacy refs.

- [ ] **Step 1: Push new canonical branches without deleting old branches**

Push `maintenance/branch-structure`, `stream1`, `cluster/production`, `kaggle/notebooks`, and `publishing/results` as new refs. Verify remote object ids with `git ls-remote --heads origin`.

- [ ] **Step 2: Open small subsystem PRs**

Open separate draft PRs targeting `main` for policy, cluster, Kaggle, and publishing. Keep `stream1` as the long-lived integration target; open a Stream 1-to-`main` release PR only for a verified release slice.

- [ ] **Step 3: Close oversized PRs only after replacement links exist**

Comment on PR #1 and PR #2 with links to the replacement PRs and preservation manifest, then close them without deleting their head branches.

- [ ] **Step 4: Enable branch protection**

Protect `main` against force pushes and deletion, require PR review and required checks. Protect `stream1` against force pushes and require Stream 1 integration checks. Do not require unavailable checks until their workflows exist and have completed successfully once.

- [ ] **Step 5: Verify GitHub state**

Use `gh api repos/TryDotAtwo/MultiGPUBeamSearch/branches` and `gh pr list --state all` to record new refs, protection status, replacement PRs, and legacy PR state.

- [ ] **Step 6: Commit the integration report**

Commit title: `docs: record branch migration integration state`

---

### Task 9: Remove only proven-redundant local worktrees and branches

**Files:**
- Modify: local Git worktree and ref metadata.
- Create: `test_results/branch_migration_cleanup_2026-08-08.md`

**Interfaces:**
- Consumes: Tasks 1–8 manifests, bundle verification, remote refs, replacement PRs.
- Produces: a smaller local branch/worktree set with full recovery paths.

- [ ] **Step 1: Remove prunable metadata only**

Run `git worktree prune --dry-run --verbose`, compare targets to the Task 1 inventory, then run `git worktree prune --verbose`. This may remove only missing `_push_final_chunk` and `_push_staged_tuner` registrations identified in the inventory.

- [ ] **Step 2: Remove clean redundant worktrees**

For each candidate, require: clean `git status`, HEAD present in the bundle, HEAD reachable from a canonical or legacy ref, and no unique SHA-manifest file. Remove one worktree at a time and rerun `git worktree list --porcelain`.

- [ ] **Step 3: Retain dirty worktrees until their files are classified**

Do not remove root `main`, CayleyPy ingest, public notebook, profile-capacity, sequence-alignment, piece-transformer, or accumulator worktrees while their Task 1 dirty entries remain unaccounted for.

- [ ] **Step 4: Delete redundant local branch names only after ancestry proof**

For each original local branch, require its exact tip to be reachable from `legacy/2026-08-08/*` and the recovery bundle. Record the command and result before `git branch -d`; use `-D` only when the branch is intentionally preserved solely by the legacy ref and bundle and the report names that exact SHA.

- [ ] **Step 5: Keep remote legacy branches for a cooling period**

Do not delete the original GitHub branches in this migration. Mark them legacy in the integration report and defer remote deletion until replacement PRs are merged and a later explicit cleanup verifies tags, pins, and external links.

- [ ] **Step 6: Run final repository integrity checks**

```powershell
git fsck --full
git worktree list --porcelain
git show-ref | Sort-Object
git bundle verify 'D:/100XH100-backups/MultiGPUBeamSearch-before-branch-migration-2026-08-08.bundle'
```

Expected: zero Git integrity errors; every retained worktree exists; every removed tip remains recoverable.

- [ ] **Step 7: Record cleanup state and commit**

Commit title: `docs: record safe legacy worktree cleanup`

---

## Self-Review

- Spec coverage: the plan preserves dirty local work, GitHub-only commits, Stream 1 MLP/Transformer/common development, cluster launchers, Kaggle notebooks, publishing services, tags, PR history, and Codex attribution.
- Destructive-action coverage: no published ref is rewritten; remote deletion is explicitly deferred; local cleanup requires SHA, bundle, ancestry, and dirty-file gates.
- Verification coverage: each subsystem has independent syntax, build, test, content-equivalence, and evidence requirements; unavailable GPU/cluster gates remain explicitly unpassed.
- Placeholder scan: all migration branches, artifacts, SHAs, commands, and acceptance conditions are concrete; no implementation placeholder remains.
