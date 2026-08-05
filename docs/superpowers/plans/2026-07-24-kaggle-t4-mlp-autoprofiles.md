# Kaggle 2xT4 MLP Autoprofiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate one private Kaggle notebook that accepts an exact user beam, exports either supported MLP checkpoint family, and automatically selects independently measured 2xT4 profiles anchored at `2**16..2**25` for output-1 and output-move-count models.

**Architecture:** Keep profile selection and validation in a small pure-Python module with deterministic unit tests. Generate the user notebook and the tuning kernel from versioned Python builders, record raw Kaggle evidence under `test_results/`, then freeze only stable measured configurations into a JSON profile registry consumed by the notebook.

**Tech Stack:** Python 3, pytest, nbformat/JSON, PyTorch checkpoint inspection, Kaggle CLI, Kaggle 2xT4, CMake/Ninja, torchrun, existing CUDA/C++ `production_runner`.

## Global Constraints

- Support only `batchnorm-folded` PilgrimAttnRes-style and `resmlp-layernorm` ResMLPDistance-style checkpoints.
- Accept only `output_dim == 1` or `output_dim == move_count`.
- Select `profile_power = clamp(round_half_up(log2(B)), 16, 25)`.
- Preserve the user's requested beam; only the existing `world_size * shard_count * alignment` rule may increase the effective beam.
- Keep output-1 and output-move-count profiles independent.
- Validate on real Kaggle 2xT4; never fake two ranks on one GPU.
- Keep all notebooks private unless the user separately approves publication.
- Do not change CUDA/C++ search algorithms as part of configuration tuning.
- Preserve unrelated dirty worktree changes.
- Store logs, outputs, and verification reports under `test_results/`.

---

### Task 1: Pure profile registry and selector

**Files:**
- Create: `tools/kaggle_t4_mlp_profiles.py`
- Create: `configs/kaggle_t4_mlp_profiles.json`
- Create: `tests/test_kaggle_t4_mlp_profiles.py`

**Interfaces:**
- Produces: `round_half_up_log2(beam_width: int) -> int`
- Produces: `select_profile(profiles: dict, beam_width: int, output_dim: int, move_count: int) -> dict`
- Produces: `align_beam(beam_width: int, world_size: int, shard_count: int, alignment: int = 1024) -> int`
- Produces: JSON schema with keys `schema_version`, `hardware`, `profiles.output1`, and `profiles.output_move_count`.

- [ ] **Step 1: Write failing selector tests**

```python
def test_non_power_beam_selects_nearest_log_anchor():
    assert round_half_up_log2(24_000_000) == 25

def test_profile_selection_does_not_replace_requested_beam():
    selected = select_profile(PROFILES, 24_000_000, output_dim=1, move_count=24)
    assert selected["requested_beam"] == 24_000_000
    assert selected["profile_power"] == 25

def test_alignment_is_the_only_beam_adjustment():
    assert align_beam(1_000_003, 2, 8, 1024) == 1_015_808

def test_output_move_count_requires_exact_match():
    with pytest.raises(ValueError, match="output_dim"):
        select_profile(PROFILES, 2**20, output_dim=12, move_count=24)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest -q tests/test_kaggle_t4_mlp_profiles.py`

Expected: collection or import failure because `tools.kaggle_t4_mlp_profiles` does not exist.

- [ ] **Step 3: Implement the selector and seed registry**

Implement half-up rounding as:

```python
power = int(math.floor(math.log2(beam_width) + 0.5))
return min(25, max(16, power))
```

Reject non-positive beams, unsupported output dimensions, missing anchors, non-2xT4 registry hardware, and profiles whose `validation_status` is neither `measured` nor `seed`.
Return requested beam, aligned beam, selected model class, anchor, and a copy of runtime parameters.

Seed all anchors with explicit `validation_status: "seed"` and existing safe evidence-based defaults. Do not label seed rows optimal or measured.

- [ ] **Step 4: Run selector tests**

Run: `pytest -q tests/test_kaggle_t4_mlp_profiles.py`

Expected: all tests pass.

- [ ] **Step 5: Commit selector**

```powershell
git add tools/kaggle_t4_mlp_profiles.py configs/kaggle_t4_mlp_profiles.json tests/test_kaggle_t4_mlp_profiles.py
git commit -m "feat: add Kaggle T4 MLP profile selector"
```

### Task 2: Checkpoint and manifest contract validation

**Files:**
- Modify: `tools/kaggle_t4_mlp_profiles.py`
- Create: `tests/test_kaggle_t4_mlp_contract.py`

**Interfaces:**
- Consumes: exporter manifests emitted by `tools/export_stream1_mlp.py`.
- Produces: `validate_manifest(manifest: dict, state_len: int, num_classes: int, move_count: int) -> str`, returning `output1` or `output_move_count`.
- Produces: `supported_model_header() -> str`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_output1_manifest_is_supported():
    assert validate_manifest(MANIFEST | {"output_dim": 1}, 120, 120, 24) == "output1"

def test_move_count_manifest_is_supported():
    assert validate_manifest(MANIFEST | {"output_dim": 24}, 120, 120, 24) == "output_move_count"

def test_arbitrary_output_is_rejected():
    with pytest.raises(ValueError, match="only output_dim=1 or output_dim=move_count"):
        validate_manifest(MANIFEST | {"output_dim": 7}, 120, 120, 24)

def test_header_names_both_checkpoint_formats():
    text = supported_model_header()
    assert "batchnorm-folded" in text
    assert "resmlp-layernorm" in text
    assert "arbitrary PyTorch" in text
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/test_kaggle_t4_mlp_contract.py`

Expected: import failure for the new functions.

- [ ] **Step 3: Implement exact validation**

Require exact equality for `state_len`, `num_classes`, and supported output mode. Include observed and expected values in every error. The header must say arbitrary PyTorch models are unsupported and name both accepted checkpoint layouts.

- [ ] **Step 4: Exercise both existing exporters**

Run:

```powershell
python tools/export_stream1_mlp.py --weights test_results/fake_resmlp_ckpt.pt --out test_results/autoprofiles_local_resmlp --format resmlp-layernorm --dtype bf16
pytest -q tests/test_kaggle_t4_mlp_contract.py::test_deterministic_batchnorm_fixture_exports
```

The focused pytest test must build a deterministic minimal BatchNorm checkpoint with `input_layer`, `bn1`, `hidden_layer`, `bn2`, one `residual_blocks.0` block, and an `output_layer` of width one in pytest's temporary directory. It then invokes `export_batchnorm_folded`, loads `manifest.json`, and calls `validate_manifest`.

Expected: both exports emit `stream1_export_done`; manifests pass `validate_manifest`.

- [ ] **Step 5: Run contract tests**

Run: `pytest -q tests/test_kaggle_t4_mlp_contract.py tests/test_kaggle_t4_mlp_profiles.py`

Expected: all tests pass.

- [ ] **Step 6: Commit contract validation**

```powershell
git add tools/kaggle_t4_mlp_profiles.py tests/test_kaggle_t4_mlp_contract.py
git commit -m "test: enforce Kaggle MLP notebook contract"
```

### Task 3: Build the private tuning kernel

**Files:**
- Create: `tools/build_kaggle_t4_mlp_autoprofile_sweep.py`
- Create: `kaggle_t4_mlp_autoprofile_sweep/kernel-metadata.json`
- Generate: `kaggle_t4_mlp_autoprofile_sweep/t4-mlp-autoprofile-sweep.ipynb`
- Create: `tests/test_build_kaggle_t4_mlp_autoprofile_sweep.py`

**Interfaces:**
- Consumes: profile seeds and both fixed benchmark checkpoints.
- Produces: notebook output `autoprofile_attempts.csv`, `selected_profiles.json`, `run_summary.json`, and full rank logs.

- [ ] **Step 1: Write failing notebook-builder tests**

Assert generated metadata is private with `enable_gpu: true`, the notebook checks exactly two T4 GPUs, runs explicit `--nproc-per-node=2`, enumerates both model classes and beam powers `16..25`, records warmup separately, writes every failed row, and never edits the requested beam before passing it to the runner.

- [ ] **Step 2: Run builder test and verify failure**

Run: `pytest -q tests/test_build_kaggle_t4_mlp_autoprofile_sweep.py`

Expected: builder import or generated-file failure.

- [ ] **Step 3: Implement bounded staged sweep**

Use a shallow-to-deep schedule:

- all anchors: correctness plus frontier-growth timing;
- anchors `16, 19, 22, 25`: full candidate matrix;
- remaining anchors: evaluate the winning neighboring configurations plus the seed;
- output-1 and output-move-count are separate rows;
- stop only an individual configuration on timeout, OOM, overflow, or invalid config;
- retain the whole kernel run and proceed to the next configuration.

Each attempt row must contain `config_id`, model class, output dimension, beam, profile power, B_MICRO, concurrency, ring slots, shards, capacity scale, Stream4 parameters, requested/effective/local beam, static/free VRAM, measured depth range, steady-state seconds, throughput, return code, status, error classification, and log path.

- [ ] **Step 4: Validate generated notebook**

Run:

```powershell
python tools/build_kaggle_t4_mlp_autoprofile_sweep.py
python -m json.tool kaggle_t4_mlp_autoprofile_sweep/t4-mlp-autoprofile-sweep.ipynb
pytest -q tests/test_build_kaggle_t4_mlp_autoprofile_sweep.py
```

Expected: JSON parses and all ordinary Python cells parse with `ast.parse`.

- [ ] **Step 5: Commit private sweep package**

```powershell
git add tools/build_kaggle_t4_mlp_autoprofile_sweep.py kaggle_t4_mlp_autoprofile_sweep tests/test_build_kaggle_t4_mlp_autoprofile_sweep.py
git commit -m "feat: add Kaggle T4 MLP autoprofile sweep"
```

### Task 4: Run and iterate the real 2xT4 sweep

**Files:**
- Modify when config-only fixes are required: `tools/build_kaggle_t4_mlp_autoprofile_sweep.py`
- Modify: `configs/kaggle_t4_mlp_profiles.json`
- Create: `test_results/kaggle_t4_mlp_autoprofiles_2026-07-24.md`
- Create: `test_results/kaggle_t4_mlp_autoprofiles_v*/`

**Interfaces:**
- Consumes: private sweep package.
- Produces: measured stable profile registry with evidence version per anchor.

- [ ] **Step 1: Push privately**

Run: `kaggle kernels push -p kaggle_t4_mlp_autoprofile_sweep`

Expected: a new private kernel version and URL.

- [ ] **Step 2: Monitor without proxy changes**

Run: `kaggle kernels status trydotatwo/cayley-beam-2xt4-mlp-autoprofiles`

Expected: `RUNNING`, then `COMPLETE` or `ERROR`. Preserve the ordinary working proxy path.

- [ ] **Step 3: Download every terminal version**

Run:

```powershell
kaggle kernels output trydotatwo/cayley-beam-2xt4-mlp-autoprofiles -p test_results/kaggle_t4_mlp_autoprofiles_latest
```

Expected: notebook log, per-rank logs, attempt CSV, selected JSON, and summary JSON are present.

- [ ] **Step 4: Diagnose only with notebook/config changes**

Classify each failure from logs. Reduce only sweep breadth, timeout, capacity, batch, concurrency, or memory settings as evidence requires. Do not alter CUDA/C++ algorithms. Repush privately until representative anchors, including `2**25`, complete for both output classes.

- [ ] **Step 5: Freeze profiles**

For every output class and anchor `16..25`, write the fastest stable measured row with margin into `configs/kaggle_t4_mlp_profiles.json`. Include kernel slug, version, timing, memory, and `validation_status: "measured"`. Never relabel an unmeasured seed.

- [ ] **Step 6: Write evidence report and run registry tests**

Run: `pytest -q tests/test_kaggle_t4_mlp_profiles.py tests/test_kaggle_t4_mlp_contract.py`

Expected: all 20 anchor entries are measured and tests pass.

- [ ] **Step 7: Commit measured profiles**

```powershell
git add configs/kaggle_t4_mlp_profiles.json test_results/kaggle_t4_mlp_autoprofiles_2026-07-24.md
git commit -m "perf: tune Kaggle 2xT4 MLP beam profiles"
```

### Task 5: Build the universal user notebook

**Files:**
- Create: `tools/build_kaggle_t4_mlp_user_notebook.py`
- Create: `kaggle_t4_mlp_universal/kernel-metadata.json`
- Generate: `kaggle_t4_mlp_universal/cayley-beam-2xt4-mlp.ipynb`
- Create: `tests/test_build_kaggle_t4_mlp_user_notebook.py`

**Interfaces:**
- Consumes: `configs/kaggle_t4_mlp_profiles.json` and `tools/kaggle_t4_mlp_profiles.py`.
- Produces: one private user notebook with first-cell configuration and the artifact contract from the design.

- [ ] **Step 1: Write failing user-notebook tests**

Tests must verify the visible header names both supported formats and rejects arbitrary PyTorch; the first code cell contains checkpoint, format, dtype, puzzle paths, exact beam, depth, puzzle ids, and mode; selector uses the registry; requested beam is passed unchanged; explicit torchrun topology is present; preflight and output artifact names are present.

- [ ] **Step 2: Run test and verify failure**

Run: `pytest -q tests/test_build_kaggle_t4_mlp_user_notebook.py`

Expected: builder import or generated-file failure.

- [ ] **Step 3: Implement the notebook builder**

The notebook must:

- copy the tested selector and measured profile registry into self-contained cells;
- expose only ordinary settings in the first code cell;
- load a local/input checkpoint without assuming a Kaggle model owner;
- export by selected supported format;
- print requested beam, anchor, effective beam, and delta;
- preflight hardware, manifest, VRAM, history, and disk;
- stream rank-zero progress and save full logs;
- write `selected_profile.json`, `run_summary.json`, `beam_run_results.csv`, and `submission.csv` in solve mode.

- [ ] **Step 4: Validate notebook locally**

Run:

```powershell
python tools/build_kaggle_t4_mlp_user_notebook.py
python -m json.tool kaggle_t4_mlp_universal/cayley-beam-2xt4-mlp.ipynb
pytest -q tests/test_build_kaggle_t4_mlp_user_notebook.py
```

Expected: all tests pass and ordinary code cells AST-parse.

- [ ] **Step 5: Commit universal notebook**

```powershell
git add tools/build_kaggle_t4_mlp_user_notebook.py kaggle_t4_mlp_universal tests/test_build_kaggle_t4_mlp_user_notebook.py
git commit -m "feat: add universal Kaggle 2xT4 MLP notebook"
```

### Task 6: Real user-notebook smoke and handoff

**Files:**
- Modify for config-only fixes: `tools/build_kaggle_t4_mlp_user_notebook.py`
- Create: `test_results/kaggle_t4_mlp_universal_2026-07-24.md`
- Create: `test_results/kaggle_t4_mlp_universal_v*/`
- Modify: `memory/CHANGELOG.md`
- Modify: `memory/PROMPTS.md`

**Interfaces:**
- Consumes: completed universal private notebook.
- Produces: verified private Kaggle URL and exact usage/evidence report.

- [ ] **Step 1: Push the universal notebook privately**

Run: `kaggle kernels push -p kaggle_t4_mlp_universal`

Expected: private kernel version launches on two T4 GPUs.

- [ ] **Step 2: Smoke both output modes**

Run one representative non-power-of-two requested beam for each output mode. Verify the selected anchor follows half-up `log2`, requested beam remains exact in runner arguments, effective beam changes only by alignment, and both ranks reach measured depths.

- [ ] **Step 3: Download and inspect artifacts**

Run:

```powershell
kaggle kernels output trydotatwo/cayley-beam-2xt4-mlp-universal -p test_results/kaggle_t4_mlp_universal_latest
```

Expected: selected profile, summary, result CSV, both rank logs, and submission in solve mode.

- [ ] **Step 4: Run final local gate**

Run:

```powershell
pytest -q tests/test_kaggle_t4_mlp_profiles.py tests/test_kaggle_t4_mlp_contract.py tests/test_build_kaggle_t4_mlp_autoprofile_sweep.py tests/test_build_kaggle_t4_mlp_user_notebook.py
git diff --check
```

Expected: all tests pass and there are no whitespace errors in task files.

- [ ] **Step 5: Update project records**

Record exact kernel slugs/versions, hardware, timings, selected parameters,
failure rows, artifact paths, and private URL in the test report and
`memory/CHANGELOG.md`. Preserve the user's requirement wording in
`memory/PROMPTS.md`.

- [ ] **Step 6: Commit final verification**

```powershell
git add test_results/kaggle_t4_mlp_universal_2026-07-24.md memory/CHANGELOG.md memory/PROMPTS.md
git commit -m "test: verify universal Kaggle 2xT4 MLP notebook"
```


## Approved Execution Refinement

- Tune every `2**16..2**25` anchor by completing and timing `depth_done=8`;
  tuning attempts are not required to solve the puzzle.
- The final notebook gate is exactly two real 2xT4 solve runs on puzzle `0`,
  requested beam `2**21`, and maximum depth `100`: one output-24 model and one
  output-1 model. Both returned paths must validate as solutions.
