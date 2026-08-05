# CayleyPy Results Publisher Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a repository-hosted Codex plugin with a dependency-free Python CLI that safely submits canonical Kaggle v1 and native/SLURM v2 CayleyPy results through Cloudflare and verifies terminal publication receipts.

**Architecture:** A single focused Python module owns parsing, deterministic batching, HTTP transport, receipt persistence, and polling. The Codex skill and two reference guides call that CLI rather than duplicating network logic. A repo marketplace exposes the plugin publicly, while all server-side schema/replay validation remains authoritative in the existing Worker.

**Tech Stack:** Python 3.10+ standard library, `unittest`, Codex plugin/skill manifests, existing Cloudflare Worker API, Markdown/JSON templates.

## Global Constraints

- Work only on `codex/cayleypy-results-ingest`; do not modify beam-search or CUDA architecture.
- Support schema v1 at `/v1/results` and schema v2 at `/v2/results`; reject mixed versions.
- Never require or embed GitHub/Cloudflare credentials.
- Each deterministic gzip request is at most 32 MiB compressed and 64 MiB decompressed; split only between envelopes.
- Pin the official public Cloudflare origin `https://cayleypy-results-ingest-staging.tupa-expert.workers.dev` and routes in the client; normal users never enter endpoints.
- Never follow redirects or trust cross-origin status URLs.
- HTTP 202 is durable acceptance, not proof of GitHub publication.
- Keep submitted envelopes and secrets out of logs and receipt manifests.
- Use only Python standard-library runtime dependencies.

---

## File structure

- Create `.agents/plugins/marketplace.json`: repo marketplace containing the public plugin entry.
- Create `plugins/cayleypy-results-publisher/.codex-plugin/plugin.json`: minimal validated plugin manifest.
- Create `plugins/cayleypy-results-publisher/skills/publish-cayleypy-results/SKILL.md`: agent routing and safety contract.
- Create `plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py`: all client behavior and CLI entry point.
- Create `plugins/cayleypy-results-publisher/tests/test_cayleypy_submit.py`: standard-library unit/integration tests.
- Create `plugins/cayleypy-results-publisher/references/HUMAN_GUIDE_RU.md`: copy-paste human guide.
- Create `plugins/cayleypy-results-publisher/references/AGENT_PROTOCOL.md`: deterministic agent protocol.
- Create `plugins/cayleypy-results-publisher/templates/kaggle-v1.json`: valid v1 batch copied from the checked-in golden fixture with explanatory safe values.
- Create `plugins/cayleypy-results-publisher/templates/native-v2.json`: valid v2 batch copied from the checked-in native example.
- Create `plugins/cayleypy-results-publisher/templates/publisher-config-kaggle.json`: fill-once common Kaggle fields.
- Create `plugins/cayleypy-results-publisher/templates/publisher-config-native.json`: fill-once common native fields.
- Create `plugins/cayleypy-results-publisher/templates/solutions.csv`: documented simple solution-table contract.
- Modify `README.md`: public installation and quick-start link.
- Modify `memory/PROMPTS.md` and `memory/CHANGELOG.md`: requirements and completed implementation evidence.
- Create `test_results/cayleypy_results_publisher_plugin_2026-08-03.md`: exact validation and staging evidence.

### Task 1: Scaffold and validate the public plugin package

**Files:**
- Create: `.agents/plugins/marketplace.json`
- Create: `plugins/cayleypy-results-publisher/.codex-plugin/plugin.json`
- Create: `plugins/cayleypy-results-publisher/skills/publish-cayleypy-results/SKILL.md`
- Test: plugin-creator and skill-creator validators

**Interfaces:**
- Produces plugin name `cayleypy-results-publisher` and skill name `publish-cayleypy-results`.
- Marketplace source path is `./plugins/cayleypy-results-publisher` with `AVAILABLE` / `ON_INSTALL` policy and `Developer Tools` category.

- [ ] **Step 1: Scaffold the repository plugin and marketplace**

Run the plugin-creator scaffold from its skill root with:

```powershell
python scripts/create_basic_plugin.py cayleypy-results-publisher `
  --path D:/100XH100/.worktrees/cayleypy-results-ingest/plugins `
  --marketplace-path D:/100XH100/.worktrees/cayleypy-results-ingest/.agents/plugins/marketplace.json `
  --with-skills --with-scripts --with-marketplace
```

Expected: normalized plugin folder, valid manifest, skill folder, scripts folder, and repo marketplace entry.

- [ ] **Step 2: Replace scaffold prose with the minimal skill contract**

The skill frontmatter must be:

```yaml
---
name: publish-cayleypy-results
description: Publish complete canonical CayleyPy Kaggle v1 or native v2 result envelopes through the anonymous Cloudflare ingest API and verify receipts without exposing secrets.
---
```

The body must route submission through `scripts/cayleypy_submit.py`, require reading `AGENT_PROTOCOL.md`, forbid fabricated metadata, and define HTTP 202 versus terminal publication.

- [ ] **Step 3: Validate the scaffold**

Run:

```powershell
python "$env:USERPROFILE/.codex/skills/.system/plugin-creator/scripts/validate_plugin.py" plugins/cayleypy-results-publisher
python "$env:USERPROFILE/.codex/skills/.system/skill-creator/scripts/quick_validate.py" plugins/cayleypy-results-publisher/skills/publish-cayleypy-results
```

Expected: both validators exit 0 with no placeholders.

- [ ] **Step 4: Commit the standalone package scaffold**

```powershell
git add .agents/plugins/marketplace.json plugins/cayleypy-results-publisher
git commit -m "Scaffold public CayleyPy publisher plugin"
```

### Task 2: Implement parsing and deterministic bounded archives with TDD

**Files:**
- Create: `plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py`
- Create: `plugins/cayleypy-results-publisher/tests/test_cayleypy_submit.py`

**Interfaces:**
- `PublisherConfig(schema_version, common, puzzle_contexts)` validates one run-level config.
- `init_config(source: Literal["kaggle", "native"], output: Path) -> None`
- `load_config(path: Path) -> PublisherConfig`
- `load_envelopes(path: Path, config: PublisherConfig | None) -> tuple[int, list[dict[str, object]]]`
- `canonical_bytes(value: object) -> bytes`
- `gzip_bytes(payload: bytes) -> bytes`
- `partition_batches(version: int, envelopes: list[dict[str, object]], max_compressed: int, max_raw: int) -> list[ArchivePart]`
- `ArchivePart(index: int, count: int, version: int, raw: bytes, compressed: bytes)`
- `expand_solution_row(config: PublisherConfig, row: dict[str, str]) -> dict[str, object]`
- `preflight(path: Path, config_path: Path | None) -> PreflightReport`

- [ ] **Step 1: Write failing parser tests**

Cover one move string, CSV, TSV, a bare envelope, versioned batch, JSONL, empty input, malformed JSON, missing config fields, mixed versions, mismatched outer/inner versions, non-object envelopes, excessive input count, and unsafe secret-like top-level fields. Assert exact stable error codes such as `INPUT_EMPTY`, `INPUT_MIXED_SCHEMA`, and `INPUT_VERSION_MISMATCH`.

- [ ] **Step 2: Run parser tests and confirm RED**

```powershell
python -m unittest discover -s plugins/cayleypy-results-publisher/tests -p "test_*.py" -v
```

Expected: import or missing-function failures.

- [ ] **Step 3: Implement bounded parsing and preflight**

Use `json.loads`, reject duplicate JSON object keys through `object_pairs_hook`, and parse CSV/TSV only with the exact header `puzzle_id,solution,final_orientation,search_mode,collection_index,collection_status,solved_depth,touch_depth,reflected_source_solution,searched_solution`. Cap input bytes/count before materializing and require integer `schema_version in {1,2}`. `publisher-config.json` contains `common` run metadata and `puzzle_contexts` keyed by puzzle ID with puzzle type, initial state, central state, and generators. Expansion replays dot-separated moves and derives length, reached-state/proof hashes, UUIDv7, timestamp, and semantic idempotency; it overlays no arbitrary fields. Return immutable report facts only: version, envelope count, raw bytes, estimated part count, pinned endpoint path.

- [ ] **Step 4: Write failing deterministic archive tests**

Assert compact sorted canonical JSON, UTF-8 preservation, gzip `mtime=0`, stable bytes across two runs, split only between envelopes, correct 32 MiB/64 MiB boundary behavior with injectable small limits, preserved ordering, and local failure `ENVELOPE_TOO_LARGE`.

- [ ] **Step 5: Implement `ArchivePart` and `partition_batches`**

Build `{schema_version: version, results: [...]}` for each candidate part. Add envelopes greedily while both raw and compressed sizes fit; finalize the previous part before adding the overflowing envelope; populate final `index` and `count` only after partitioning.

- [ ] **Step 6: Run Task 2 tests and commit**

```powershell
python -m unittest discover -s plugins/cayleypy-results-publisher/tests -p "test_*.py" -v
git add plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py plugins/cayleypy-results-publisher/tests/test_cayleypy_submit.py
git commit -m "Add deterministic CayleyPy result archive builder"
```

Expected: all parser/archive tests pass.

### Task 3: Implement safe submission, receipts, resume, and polling with TDD

**Files:**
- Modify: `plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py`
- Modify: `plugins/cayleypy-results-publisher/tests/test_cayleypy_submit.py`

**Interfaces:**
- `OFFICIAL_ENDPOINT_BASE: Final[str] = "https://cayleypy-results-ingest-staging.tupa-expert.workers.dev"`
- `submit_parts(parts: list[ArchivePart], config: SubmitConfig, transport: Transport) -> ReceiptManifest`
- `poll_manifest(manifest: ReceiptManifest, config: PollConfig, transport: Transport) -> PollSummary`
- `save_manifest(path: Path, manifest: ReceiptManifest) -> None`
- `load_manifest(path: Path) -> ReceiptManifest`
- CLI subcommands: `init`, `preflight`, `submit`, `poll`.

- [ ] **Step 1: Write failing transport tests using a local HTTP server**

Test v1/v2 paths, `Content-Type: application/gzip`, archive index/count headers, sequential requests, no redirects, bounded response reads, HTTP 202 parsing, partial `errors[]`, malformed JSON, `429 Retry-After`, retryable 5xx, non-retryable 4xx, and absence of payload markers in stdout/stderr.

- [ ] **Step 2: Implement an injectable standard-library transport**

Use `urllib.request` with a redirect-rejecting handler, explicit connect/read timeout, bounded response reading, and the pinned HTTPS origin. Permit `http://127.0.0.1` only through an undocumented test/development flag and inject retry scheduling for deterministic tests.

- [ ] **Step 3: Write failing receipt persistence/resume tests**

Assert atomic temp-file replacement, schema-versioned manifest shape, receipt deduplication by submission ID, no envelope contents, preservation after part 1 succeeds and part 2 fails, and resume skipping already accepted part digests.

- [ ] **Step 4: Implement receipt manifests and resumable submission**

Manifest fields are limited to client version, endpoint origin, input SHA-256, archive part digests/statuses, submission IDs, idempotency keys, status URLs, timestamps, and terminal safe statuses. Save after every accepted response before starting the next part.

- [ ] **Step 5: Write failing poll tests**

Cover same-origin status URLs, `received/validated/queued/publishing` nonterminal states, `published/duplicate` success, `rejected/failed` failure, unknown states, timeout, and mixed summary counts. Reject cross-origin and non-`/v1/submissions/<uuid>` URLs.

- [ ] **Step 6: Implement polling and CLI exit codes**

Return 0 only for successful preflight, complete acceptance without `--wait`, or all-terminal successful polling. Return distinct non-zero codes for local validation, transport/partial acceptance, terminal rejection, and timeout/unresolved states. Print one final JSON summary with no payload data.

- [ ] **Step 7: Run Task 3 tests and commit**

```powershell
python -m unittest discover -s plugins/cayleypy-results-publisher/tests -p "test_*.py" -v
git add plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py plugins/cayleypy-results-publisher/tests/test_cayleypy_submit.py
git commit -m "Add resilient anonymous result submission client"
```

### Task 4: Add templates and human/agent documentation

**Files:**
- Create: `plugins/cayleypy-results-publisher/templates/kaggle-v1.json`
- Create: `plugins/cayleypy-results-publisher/templates/native-v2.json`
- Create: `plugins/cayleypy-results-publisher/templates/publisher-config-kaggle.json`
- Create: `plugins/cayleypy-results-publisher/templates/publisher-config-native.json`
- Create: `plugins/cayleypy-results-publisher/templates/solutions.csv`
- Create: `plugins/cayleypy-results-publisher/references/HUMAN_GUIDE_RU.md`
- Create: `plugins/cayleypy-results-publisher/references/AGENT_PROTOCOL.md`
- Modify: `plugins/cayleypy-results-publisher/skills/publish-cayleypy-results/SKILL.md`
- Modify: `README.md`

**Interfaces:**
- Human quick-start commands invoke `init`, `submit --wait`, and `poll --manifest` exactly as implemented; they do not ask for an endpoint.
- Agent protocol defines inputs, refusal conditions, command sequence, and final report fields.

- [ ] **Step 1: Generate valid templates from canonical fixtures**

Copy one v1 envelope from `configs/cayleypy_results_v1_golden.json` into a versioned batch and copy `configs/cayleypy_results_v2_example_payload.json` as v2. Add strict fill-once Kaggle/native configs and a solutions CSV whose fields map exactly to the expansion allowlist. Do not add comments inside JSON or weaken canonical examples into placeholders that fail server validation.

- [ ] **Step 2: Write the Russian human guide**

Lead with: install, run `init`, fill one config, provide a move/CSV/JSON, run `submit --wait`. Include the fixed official endpoint, zero-token explanation, every config/table field, Kaggle best/all-results recipes, cluster/local recipes, dry-run, resume, terminal interpretation, GitHub location, troubleshooting table, and Windows/Linux/Kaggle command variants.

- [ ] **Step 3: Write the agent protocol and finish the skill**

Use imperative steps and explicit fail-closed rules. Require truthful provenance, local preflight, sequential submission, receipt preservation, polling before success claims, and an exact summary containing version, envelope/part counts, receipt IDs, terminal counts, manifest path, and unresolved errors.

- [ ] **Step 4: Add README discovery and test every documented command**

Run both templates through `preflight`; run mocked `submit` and `poll` examples from the guide. Search docs for stale command names, placeholders, private paths, tokens, and unsupported claims.

- [ ] **Step 5: Validate and commit documentation**

```powershell
python "$env:USERPROFILE/.codex/skills/.system/plugin-creator/scripts/validate_plugin.py" plugins/cayleypy-results-publisher
python "$env:USERPROFILE/.codex/skills/.system/skill-creator/scripts/quick_validate.py" plugins/cayleypy-results-publisher/skills/publish-cayleypy-results
git add README.md plugins/cayleypy-results-publisher .agents/plugins/marketplace.json
git commit -m "Document Kaggle and native result publication"
```

### Task 5: Full regression, live staging smoke, evidence, and publication

**Files:**
- Modify: `memory/PROMPTS.md`
- Modify: `memory/CHANGELOG.md`
- Create: `test_results/cayleypy_results_publisher_plugin_2026-08-03.md`

**Interfaces:**
- Consumes the completed plugin, CLI, templates, existing Worker tests, and staging endpoint.
- Produces exact reproducibility evidence and a pushed plugin branch; does not merge or deploy production.

- [ ] **Step 1: Run local plugin and Worker gates**

```powershell
python -m unittest discover -s plugins/cayleypy-results-publisher/tests -v
python "$env:USERPROFILE/.codex/skills/.system/plugin-creator/scripts/validate_plugin.py" plugins/cayleypy-results-publisher
Push-Location services/cayleypy-results-ingest
npm test
npm run typecheck
Pop-Location
```

Expected: all CLI/plugin tests, current schema/config tests, Worker tests, and typecheck pass.

- [ ] **Step 2: Run exact dry-runs and public clean scan**

Preflight both checked-in templates. Scan plugin/docs/manifests for local absolute paths, bearer tokens, GitHub tokens, Cloudflare API tokens, envelope dumps, unfinished placeholder markers, and misleading claims that HTTP 202 equals publication.

- [ ] **Step 3: Run one idempotent staging smoke per schema**

Submit the checked-in valid v1 and v2 examples to the staging Worker with `--wait`, saving separate manifests under `test_results/`. Reuse deterministic examples so repeated smoke runs converge as duplicates. Verify every status URL is same-origin and reaches `published` or `duplicate`, then verify the corresponding immutable GitHub staging paths.

- [ ] **Step 4: Record evidence and final project memory**

Write exact commit, Python/Node versions, commands, test counts, request part sizes, receipt IDs, terminal statuses, GitHub paths, and clean-scan result to `test_results/cayleypy_results_publisher_plugin_2026-08-03.md`. Update `memory/CHANGELOG.md` and `memory/PROMPTS.md` without secrets.

- [ ] **Step 5: Final deterministic verification and commit**

```powershell
git diff --check
git status --short
git add memory/PROMPTS.md memory/CHANGELOG.md test_results/cayleypy_results_publisher_plugin_2026-08-03.md
git commit -m "Verify public CayleyPy result publisher plugin"
```

- [ ] **Step 6: Push the Cloudflare branch and report installation links**

```powershell
git push origin codex/cayleypy-results-ingest
```

Verify the remote SHA equals local HEAD. Report the repository plugin path, human guide, agent protocol, exact tests, staging receipts, and Codex marketplace View/Share links. Keep the worktree for review.
