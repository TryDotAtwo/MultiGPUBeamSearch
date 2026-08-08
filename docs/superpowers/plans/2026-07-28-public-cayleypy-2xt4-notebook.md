# Public CayleyPy 2xT4 Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publicly release a thin copy-and-run Kaggle notebook that runs the existing MultiGPUBeamSearch on standard CayleyPy inputs with one supported MLP checkpoint, two T4 GPUs, reflection, bounded solution collection, configurable touch BFS, and best-effort result publication.

**Architecture:** Keep the notebook declarative and move stable behavior into a focused `tools/cayleypy_public/` Python package. The package validates CayleyPy data and model contracts, selects the measured 2xT4 profile without replacing the user's beam, orchestrates the existing `production_runner`, validates every path on CPU, and emits bounded publication envelopes. The notebook builder embeds only documentation, the user config, and calls into the pinned repository revision.

**Tech Stack:** Python 3.12, PyTorch checkpoint inspection, pandas, pytest, IPython notebook parsing, CMake/Ninja/CUDA/CUTLASS/NCCL, torchrun, Kaggle CLI.

## Global Constraints

- Support only the standard CayleyPy `puzzle_info.json`, `test.csv`, and `sample_submission.csv` structure.
- Require exactly two NVIDIA T4 GPUs at runtime.
- Expose only checkpoint input; auto-detect only `batchnorm-folded` and `resmlp-layernorm` MLP schemas.
- Infer `fp16` automatically on T4; do not expose a dtype setting.
- Accept only `output_dim=1` or `output_dim=move_count`.
- Treat `PUZZLE_ID_START..PUZZLE_ID_END` as inclusive and fail on every missing id.
- Preserve requested beam; apply only documented distributed alignment and select the nearest measured `2**16..2**25` profile.
- Expose `REFLECT_MODE=off|after_original|only`, `SOLUTION_MODE=first|collect`, `COLLECT_UNTIL_DEPTH`, `MAX_COLLECTED_SOLUTIONS`, and `TOUCH_BFS_RADIUS`.
- Keep experimental K2 disabled.
- Do not change Stream2/3/4/5 algorithms merely to package the public notebook.
- Result publication is best-effort and must never invalidate a successful solve.
- Do not publish secrets, checkpoint tensors, private filesystem roots, or process environments.

---

## File Structure

- `tools/cayleypy_public/__init__.py`: stable package exports only.
- `tools/cayleypy_public/config.py`: typed user configuration and fail-closed validation.
- `tools/cayleypy_public/data.py`: standard CayleyPy file loading and inclusive-range validation.
- `tools/cayleypy_public/model.py`: checkpoint family detection, export invocation, and manifest validation.
- `tools/cayleypy_public/profile.py`: registry selection plus move-count-aware capacity derivation.
- `tools/cayleypy_public/paths.py`: move replay, inversion, reflection, and solution deduplication.
- `tools/cayleypy_public/runner.py`: torchrun command/environment construction and first/collect orchestration.
- `tools/cayleypy_public/results.py`: artifacts, result envelopes, redaction, and best-effort HTTP client.
- `tools/run_cayleypy_public.py`: one CLI entrypoint used by the notebook.
- `tools/build_public_cayleypy_notebook.py`: deterministic public notebook and metadata generator.
- `configs/cayleypy_results_schema_v1.json`: notebook-side result-envelope JSON Schema.
- `kaggle_public_cayleypy_2xt4/`: generated notebook package.
- `tests/cayleypy_public/`: focused unit/integration tests.

### Task 1: Typed Configuration and Standard CayleyPy Data Contract

**Files:**
- Create: `tools/cayleypy_public/__init__.py`
- Create: `tools/cayleypy_public/config.py`
- Create: `tools/cayleypy_public/data.py`
- Create: `tests/cayleypy_public/test_config.py`
- Create: `tests/cayleypy_public/test_data.py`

**Interfaces:**
- Produces: `PublicRunConfig.from_mapping(values: Mapping[str, object]) -> PublicRunConfig`
- Produces: `load_puzzle_contract(puzzle_info_path: Path, test_csv: Path, sample_submission_csv: Path, start: int, end: int) -> PuzzleContract`
- `PuzzleContract` exposes `central_state`, `generators`, `move_names`, `move_count`, `state_len`, `num_classes`, `initial_states`, and `sample_submission`.

- [ ] **Step 1: Write failing config tests**

```python
from tools.cayleypy_public.config import PublicRunConfig

BASE = {
    "author_name": "alice", "checkpoint_path": "/kaggle/input/m/model.pth",
    "puzzle_info_json": "/kaggle/input/c/puzzle_info.json",
    "test_csv": "/kaggle/input/c/test.csv",
    "sample_submission_csv": "/kaggle/input/c/sample_submission.csv",
    "puzzle_id_start": 7, "puzzle_id_end": 9, "beam_width": 2**21,
    "max_depth": 100, "reflect_mode": "off", "reflect_source_csv": None,
    "solution_mode": "first", "collect_until_depth": 100,
    "max_collected_solutions": 1000, "touch_bfs_radius": 4,
    "publish_results": True, "results_ingest_url": "https://results.example/",
}

def test_config_accepts_exact_public_contract():
    cfg = PublicRunConfig.from_mapping(BASE)
    assert cfg.puzzle_ids == (7, 8, 9)
    assert cfg.model_dtype == "fp16"

def test_collect_depth_cannot_exceed_max_depth():
    values = {**BASE, "solution_mode": "collect", "collect_until_depth": 101}
    with pytest.raises(ValueError, match="COLLECT_UNTIL_DEPTH"):
        PublicRunConfig.from_mapping(values)

def test_only_requires_reflection_source():
    with pytest.raises(ValueError, match="REFLECT_SOURCE_CSV"):
        PublicRunConfig.from_mapping({**BASE, "reflect_mode": "only"})
```

- [ ] **Step 2: Run config tests and verify RED**

Run: `python -m pytest tests/cayleypy_public/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: tools.cayleypy_public`.

- [ ] **Step 3: Implement the frozen dataclass and enum/range gates**

```python
@dataclass(frozen=True)
class PublicRunConfig:
    author_name: str
    checkpoint_path: Path
    puzzle_info_json: Path
    test_csv: Path
    sample_submission_csv: Path
    puzzle_id_start: int
    puzzle_id_end: int
    beam_width: int
    max_depth: int
    reflect_mode: Literal["off", "after_original", "only"]
    reflect_source_csv: Path | None
    solution_mode: Literal["first", "collect"]
    collect_until_depth: int
    max_collected_solutions: int
    touch_bfs_radius: int
    publish_results: bool
    results_ingest_url: str
    model_dtype: Literal["fp16"] = "fp16"

    @property
    def puzzle_ids(self) -> tuple[int, ...]:
        return tuple(range(self.puzzle_id_start, self.puzzle_id_end + 1))
```

Validate positive beam/depth/capacity, `0 <= touch_bfs_radius <= 12`, inclusive non-empty range, exact enum allowlists, `collect_until_depth <= max_depth`, and the reflection-source rule.

- [ ] **Step 4: Write failing CayleyPy schema/range tests**

Build temporary fixtures with `central_state`, two named permutations, test ids `7,8,9`, and submission rows. Assert derived move/state/class dimensions. Assert missing id 8 and duplicate id 8 each fail with explicit messages.

- [ ] **Step 5: Implement `PuzzleContract` and strict loaders**

```python
@dataclass(frozen=True)
class PuzzleContract:
    central_state: tuple[int, ...]
    generators: dict[str, tuple[int, ...]]
    initial_states: dict[int, tuple[int, ...]]
    sample_submission: pd.DataFrame
    state_len: int
    num_classes: int

    @property
    def move_names(self) -> tuple[str, ...]:
        return tuple(self.generators)

    @property
    def move_count(self) -> int:
        return len(self.generators)
```

Require every permutation and state to have `state_len`, every permutation to equal `range(state_len)` as a set, every selected state id exactly once, and submission ids to include the selected range.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/cayleypy_public/test_config.py tests/cayleypy_public/test_data.py -q`
Expected: PASS.

```bash
git add tools/cayleypy_public tests/cayleypy_public/test_config.py tests/cayleypy_public/test_data.py
git commit -m "feat: add public CayleyPy input contract"
```

### Task 2: Automatic Supported-MLP Detection and Export

**Files:**
- Modify: `tools/export_stream1_mlp.py`
- Create: `tools/cayleypy_public/model.py`
- Create: `tests/cayleypy_public/test_model.py`

**Interfaces:**
- Produces: `detect_checkpoint_format(path: Path) -> Literal["batchnorm-folded", "resmlp-layernorm"]`
- Produces: `export_checkpoint(path: Path, out_dir: Path, num_classes: int) -> ExportedModel`
- `ExportedModel` exposes `format`, `dtype`, `checkpoint_sha256`, and sanitized `manifest`.

- [ ] **Step 1: Write failing detection tests with minimal tensor fixtures**

```python
def test_detects_batchnorm_folded(tmp_path):
    path = write_checkpoint(tmp_path, {
        "input_layer.weight": torch.zeros(8, 12),
        "hidden_layer.weight": torch.zeros(8, 8),
        "output_layer.weight": torch.zeros(1, 8),
        "bn1.running_mean": torch.zeros(8),
        "residual_blocks.0.fc1.weight": torch.zeros(8, 8),
    })
    assert detect_checkpoint_format(path) == "batchnorm-folded"

def test_rejects_mixed_schema(tmp_path):
    path = write_checkpoint(tmp_path, {
        "input_layer.weight": torch.zeros(8, 12),
        "embedding.weight": torch.zeros(12, 16),
        "head.weight": torch.zeros(1, 8),
    })
    with pytest.raises(ValueError, match="ambiguous"):
        detect_checkpoint_format(path)
```

Add ResMLP and unknown-schema cases.

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/cayleypy_public/test_model.py -q`
Expected: FAIL because `model.py` does not exist.

- [ ] **Step 3: Implement signature-set detection without trial export**

```python
BN_REQUIRED = frozenset({
    "input_layer.weight", "hidden_layer.weight", "output_layer.weight",
    "bn1.running_mean", "bn2.running_mean",
})
LN_REQUIRED = frozenset({
    "embedding.weight", "input_stack.0.weight", "input_stack.1.weight",
    "input_stack.3.weight", "head.weight",
})
```

Normalize `_orig_mod.` prefixes through the existing helpers. A schema matches only when all required keys and at least one contiguous residual block key are present. Zero or two matches fail closed.

- [ ] **Step 4: Add `--format auto` to the existing exporter**

Extend argparse choices to `auto`, call `detect_checkpoint_format`, then dispatch to the unchanged existing exporters. Ensure `source_weights` in the public sanitized manifest is only the checkpoint basename, not the absolute path.

- [ ] **Step 5: Implement export wrapper and manifest gates**

Call the exporter in-process, force `dtype="fp16"`, hash the checkpoint in 8 MiB chunks, load `manifest.json`, require puzzle `state_len/num_classes`, normalization allowlist, and `output_dim in {1, move_count}`.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/cayleypy_public/test_model.py tests/test_kaggle_t4_mlp_contract.py -q`
Expected: PASS.

```bash
git add tools/export_stream1_mlp.py tools/cayleypy_public/model.py tests/cayleypy_public/test_model.py
git commit -m "feat: auto-detect supported Stream1 MLP checkpoints"
```

### Task 3: Profile Selection and Move-Count-Aware Preflight

**Files:**
- Modify: `tools/kaggle_t4_mlp_profiles.py`
- Create: `tools/cayleypy_public/profile.py`
- Create: `tests/cayleypy_public/test_profile.py`

**Interfaces:**
- Consumes: `select_profile(...)` and checked-in `configs/kaggle_t4_mlp_profiles.json`.
- Produces: `derive_runtime(profile, beam_width, output_dim, move_count, world_size=2) -> RuntimePlan`.

- [ ] **Step 1: Write failing dynamic-capacity tests**

Assert output 1 uses `parent_batch=b_micro//move_count`; output `move_count` uses `parent_batch=b_micro`; Stream3 batch is `parent_batch*move_count*ring_slots`; shard capacity is aligned to 1024 and at least Stream3 batch, Stream4 batch, and trigger. Include move counts 18 and 24.

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/cayleypy_public/test_profile.py -q`
Expected: FAIL with missing `derive_runtime`.

- [ ] **Step 3: Implement immutable runtime plan**

```python
@dataclass(frozen=True)
class RuntimePlan:
    requested_beam: int
    effective_beam: int
    alignment_delta: int
    profile_power: int
    model_class: str
    local_beam: int
    parent_batch: int
    stream3_batch_candidates: int
    shard_capacity_candidates: int
    runtime: dict[str, int]
    cross_puzzle_profile_note: str
```

Use measured registry values, exact requested beam, and `cross_puzzle_profile_note="measured_24_move_seed"` whenever `move_count != 24`.

- [ ] **Step 4: Add preflight serialization and guard tests**

Assert JSON contains profile evidence version, hardware, requested/effective beam, alignment, actual move count, capacity derivation, history budgets, and `/tmp` free disk. Unknown validation status fails.

- [ ] **Step 5: Run all profile tests and commit**

Run: `python -m pytest tests/cayleypy_public/test_profile.py tests/test_kaggle_t4_mlp_profiles.py -q`
Expected: PASS.

```bash
git add tools/kaggle_t4_mlp_profiles.py tools/cayleypy_public/profile.py tests/cayleypy_public/test_profile.py
git commit -m "feat: derive safe public 2xT4 runtime plans"
```

### Task 4: Path Replay, Reflection, and Deduplication

**Files:**
- Create: `tools/cayleypy_public/paths.py`
- Create: `tests/cayleypy_public/test_paths.py`

**Interfaces:**
- Produces: `apply_path(state, path, generators) -> tuple[int, ...]`
- Produces: `invert_path(path: str, generators: Mapping[str, Sequence[int]]) -> str`
- Produces: `make_reflected_state(central, original_solution, generators) -> tuple[int, ...]`
- Produces: `validate_original_solution(initial, central, path, generators) -> bool`
- Produces: `SolutionRecord` and `deduplicate_solutions(records) -> list[SolutionRecord]`.

- [ ] **Step 1: Write failing algebraic tests**

Use a three-element cycle with explicit inverse names. Assert apply/invert roundtrip, reflection roundtrip, invalid move rejection, and reflected candidate inversion solves the original.

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/cayleypy_public/test_paths.py -q`
Expected: FAIL with missing module.

- [ ] **Step 3: Implement move inversion from permutations**

Do not assume `-MOVE` naming. Compute inverse permutations and require every inverse permutation to map uniquely to an existing named generator; otherwise reflection modes fail preflight with the missing move name.

- [ ] **Step 4: Implement canonical solution records and dedupe**

```python
@dataclass(frozen=True)
class SolutionRecord:
    puzzle_id: int
    variant: Literal["original", "reflected"]
    path: str
    original_oriented_path: str
    found_depth: int
    touch_depth: int
    source_solution_sha256: str | None
    valid: bool
```

Deduplicate by `(puzzle_id, sha256(original_oriented_path), sha256(reached_state))`; choose the earliest provenance deterministically.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/cayleypy_public/test_paths.py -q`
Expected: PASS.

```bash
git add tools/cayleypy_public/paths.py tests/cayleypy_public/test_paths.py
git commit -m "feat: add CayleyPy reflection and path validation"
```

### Task 5: Existing Runner Orchestration for First and Collect Modes

**Files:**
- Create: `tools/cayleypy_public/runner.py`
- Create: `tools/run_cayleypy_public.py`
- Create: `tests/cayleypy_public/test_runner.py`
- Create: `tests/cayleypy_public/fixtures/fake_production_runner.py`

**Interfaces:**
- Consumes: `PublicRunConfig`, `PuzzleContract`, `ExportedModel`, `RuntimePlan`.
- Produces: `run_public_search(...) -> RunArtifacts`.
- `RunArtifacts` exposes solution records, combined/per-rank log paths, return codes, timing summaries, and collection status.

- [ ] **Step 1: Write failing command/environment tests**

Assert `torch.distributed.run --nproc-per-node=2`, original `BEAM_WIDTH` CLI argument, rank log redirection, exact runtime env, `BEAM_SOLVED_NEIGHBORHOOD_RADIUS`, K2 radius 0, and capacity values. In collect mode assert `BEAM_SOLVE_BUCKET_MODE=1`, `BEAM_SOLVE_BUCKET_STOP_DEPTH=COLLECT_UNTIL_DEPTH`, and `BEAM_SOLVE_BUCKET_MAX_SOLUTIONS=MAX_COLLECTED_SOLUTIONS`. Assert no inherited `WORLD_SIZE/RANK/LOCAL_RANK`.

- [ ] **Step 2: Write failing first/collect parser tests**

The fake runner emits first-solution lines or a solve-bucket TSV. Assert `first` stops on one valid result. Assert `collect` sets:

```python
{
  "BEAM_SOLVE_BUCKET_MODE": "1",
  "BEAM_SOLVE_BUCKET_STOP_DEPTH": str(collect_until_depth),
  "BEAM_SOLVE_BUCKET_MAX_SOLUTIONS": str(max_collected_solutions),
  "BEAM_SOLVED_RESULT_CAPACITY": str(max_collected_solutions),
  "BEAM_SOLVE_BUCKET_RESULT_TSV": str(result_tsv),
}
```

The helper must preserve a partial TSV and report `capacity_reached` rather than treating the configured solution limit as a solver failure.

- [ ] **Step 3: Run and verify RED**

Run: `python -m pytest tests/cayleypy_public/test_runner.py -q`
Expected: FAIL with missing runner module.

- [ ] **Step 4: Implement command/env construction and log capture**

Use a unique rendezvous id/port per puzzle, `/tmp/beam_history_public/<run_id>/<puzzle>/<variant>`, `--redirects=3`, and `--tee=0:3`. Stream rank 0 while retaining both rank logs.

- [ ] **Step 5: Implement mode orchestration**

For `off`, run originals. For `after_original`, use every validated original source selected by mode, synthesize a temporary standard test row, run reflected search, invert, and revalidate. For `only`, validate every supplied source path before any GPU launch. In all modes, update submission with the shortest valid original-oriented path.

- [ ] **Step 6: Implement bounded collection semantics**

Add host-only controls `BEAM_SOLVE_BUCKET_STOP_DEPTH` and `BEAM_SOLVE_BUCKET_MAX_SOLUTIONS` to `production_runner.cu`; do not change Stream algorithms. Stop after the requested completed depth, or after deterministically writing the configured number of unique records. Emit `collection_status=depth_reached|capacity_reached`. Keep `BEAM_SOLVED_RESULT_CAPACITY` large enough for one depth and treat a true device snapshot overflow as a configuration error distinct from the successful total-solution limit. Add focused C++/CLI tests for both stop conditions.

- [ ] **Step 7: Run tests and commit**

Run: `python -m pytest tests/cayleypy_public/test_runner.py tests/cayleypy_public/test_paths.py -q`
Expected: PASS.

```bash
git add tools/cayleypy_public/runner.py tools/run_cayleypy_public.py tests/cayleypy_public/test_runner.py tests/cayleypy_public/fixtures/fake_production_runner.py
git commit -m "feat: orchestrate public CayleyPy search modes"
```

### Task 6: Result Envelope and Best-Effort Publisher

**Files:**
- Create: `configs/cayleypy_results_schema_v1.json`
- Create: `tools/cayleypy_public/results.py`
- Create: `tests/cayleypy_public/test_results.py`

**Interfaces:**
- Produces: `build_result_envelope(context, solution) -> dict[str, object]`
- Produces: `publish_results(url: str, envelopes: Sequence[dict], timeout_s: float = 15.0) -> PublishStatus`.

- [ ] **Step 1: Write failing schema/redaction/idempotency tests**

Assert required author, Kaggle provenance, proof bundle, solution, profile, model hash/manifest, hardware, timings, solver commit, and schema version. Assert the same semantic result yields the same idempotency key. Assert absolute checkpoint paths, tokens, environment keys, tensors, and unknown fields never appear.

- [ ] **Step 2: Write failing HTTP behavior tests**

Mock `202`, duplicate `200`, timeout, DNS error, `429`, and `500`. Every network failure returns `PublishStatus(ok=False, retryable=..., safe_error=...)` and never raises into the solve result.

- [ ] **Step 3: Implement bounded canonical envelopes**

Use canonical JSON (`sort_keys=True`, UTF-8, compact separators), UUIDv7 submission ids, SHA-256 idempotency, maximum 256 KiB per envelope, maximum 100 envelopes per request, and exact JSON Schema validation before POST.

- [ ] **Step 4: Implement safe publisher and durable local status**

POST `{"schema_version":1,"results":[...]}`. Write `publish_status.json` before returning. Redact URL credentials and never include response headers/body beyond a 2 KiB sanitized message.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/cayleypy_public/test_results.py -q`
Expected: PASS.

```bash
git add configs/cayleypy_results_schema_v1.json tools/cayleypy_public/results.py tests/cayleypy_public/test_results.py
git commit -m "feat: publish validated CayleyPy result envelopes"
```

### Task 7: Thin Public Notebook Builder

**Files:**
- Create: `tools/build_public_cayleypy_notebook.py`
- Create: `tests/cayleypy_public/test_notebook.py`
- Create: `kaggle_public_cayleypy_2xt4/kernel-metadata.json`
- Generate: `kaggle_public_cayleypy_2xt4/cayleypy-2xt4-universal.ipynb`

**Interfaces:**
- Produces: `build_notebook(out_dir: Path) -> tuple[Path, Path]`.

- [ ] **Step 1: Write failing notebook contract tests**

Assert the first code cell contains exactly the public settings and no `MODEL_SOURCE`, `MODEL_DTYPE`, or `CHECKPOINT_FORMAT`. Assert the header names the two supported models, the fixed `1 <= state_len <= 120` `State128` limit, the `output_dim=1` or exact-`move_count` heads, standard CayleyPy scope, two T4 requirement, all reflection/collection modes, touch BFS, best-effort publication, and claimed author semantics.

- [ ] **Step 2: Write parse/secret/public-metadata tests**

Assert JSON parse, IPython transform plus AST parse for every code cell, `is_private=False`, no tokens/private Windows paths, and no outputs/execution counts. Assert notebook pins `SOLVER_GIT_REV` to a full 40-character commit.

- [ ] **Step 3: Run and verify RED**

Run: `python -m pytest tests/cayleypy_public/test_notebook.py -q`
Expected: FAIL with missing builder.

- [ ] **Step 4: Implement five-cell notebook**

Cells: documentation, config, checkout/preflight, build/run, artifact/publish summary. The run cell invokes:

```python
checked([
    "python3", REPO / "tools/run_cayleypy_public.py",
    "--config-json", WORK / "public_run_config.json",
    "--output-dir", WORK,
])
```

Metadata uses a new public slug, exactly one T4x2 machine shape, Internet enabled, and no embedded competition/model source that prevents copying to another CayleyPy competition.

- [ ] **Step 5: Generate, test, and commit**

Run: `python tools/build_public_cayleypy_notebook.py`
Run: `python -m pytest tests/cayleypy_public/test_notebook.py -q`
Expected: PASS.

```bash
git add tools/build_public_cayleypy_notebook.py tests/cayleypy_public/test_notebook.py kaggle_public_cayleypy_2xt4
git commit -m "feat: add public CayleyPy 2xT4 notebook"
```

### Task 8: Local Integration and Regression Gate

**Files:**
- Create: `tests/cayleypy_public/test_end_to_end.py`
- Create: `test_results/public_cayleypy_local_gate_2026-07-28.md`
- Modify: `memory/CHANGELOG.md`

**Interfaces:**
- Consumes all notebook helper contracts.
- Produces a deterministic CPU/fake-runner acceptance artifact.

- [ ] **Step 1: Add an end-to-end fixture**

Use a small Cayley group, both supported checkpoint fixtures, fake rank logs, original/reflected results, a collection bucket, and a mock ingest server. Assert final submission, all solution artifacts, profile/preflight JSON, both rank logs, and publish receipt.

- [ ] **Step 2: Run focused and full Python gates**

Run: `python -m pytest tests/cayleypy_public -q`
Run: `python -m pytest tests/test_kaggle_t4_mlp_profiles.py tests/test_kaggle_t4_mlp_contract.py tests/test_build_kaggle_t4_mlp_autoprofile_sweep.py tests/test_build_kaggle_t4_mlp_user_notebook.py -q`
Expected: all PASS.

- [ ] **Step 3: Run static notebook/public scan**

Run a script that checks JSON, transformed AST, forbidden strings (`sk-`, `ghp_`, `github_pat_`, `GITHUB_TOKEN=`, local usernames, private absolute paths), metadata visibility, and exact first-cell settings. Save command output in the report.

- [ ] **Step 4: Record evidence and commit**

```bash
git add tests/cayleypy_public test_results/public_cayleypy_local_gate_2026-07-28.md memory/CHANGELOG.md
git commit -m "test: validate public CayleyPy notebook locally"
```

### Task 9: Kaggle Staging, End-to-End Validation, and Public Release

**Files:**
- Modify: `kaggle_public_cayleypy_2xt4/kernel-metadata.json`
- Create: `test_results/kaggle_public_cayleypy_2xt4_release/`
- Create: `test_results/kaggle_public_cayleypy_release_2026-07-28.md`
- Modify: `memory/CHANGELOG.md`

**Interfaces:**
- Requires the production `RESULTS_INGEST_URL` from the second implementation plan.
- Produces a public Kaggle kernel URL and a verified Cloudflare/GitHub result receipt.

- [ ] **Step 1: Push a private staging version**

Set `is_private=true`, use bounded `2**16`, depth 8, one puzzle, `REFLECT_MODE=off`, `SOLUTION_MODE=first`, and publication to staging ingest. Push with ordinary Kaggle CLI.

- [ ] **Step 2: Monitor and download terminal artifacts**

Run `kaggle kernels status trydotatwo/cayleypy-2xt4-universal-beam`. On terminal state download all outputs under `test_results/kaggle_public_cayleypy_2xt4_release/`. Verify two T4s, both rank logs, model auto-detection, profile/capacity, CPU path validation or clean bounded no-solution status, and ingest receipt/status lookup.

- [ ] **Step 3: Run final public-artifact secret and path scan**

Scan source notebook, executed notebook log, all JSON/CSV, and rendered HTML text for tokens, Cloudflare/GitHub secret material, private local paths, and unbounded environments. Any hit blocks publication.

- [ ] **Step 4: Switch metadata to public and push exactly the reviewed source state**

Set `is_private=false`, regenerate without outputs, confirm the notebook source hash equals the reviewed artifact, then `kaggle kernels push -p kaggle_public_cayleypy_2xt4`.

- [ ] **Step 5: Verify the public version and record release evidence**

Open the public URL, confirm copyability, rendered header, first config cell, public visibility, output artifacts, and service publication status. Record exact kernel version, timings, hashes, helper commit, ingest submission ids, and results-repository URLs.

- [ ] **Step 6: Commit release notes**

```bash
git add kaggle_public_cayleypy_2xt4/kernel-metadata.json test_results/kaggle_public_cayleypy_release_2026-07-28.md memory/CHANGELOG.md
git commit -m "docs: record public CayleyPy notebook release"
```