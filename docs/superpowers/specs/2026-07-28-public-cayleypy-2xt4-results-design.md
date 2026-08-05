# Public CayleyPy 2xT4 Solver and Results Ingest Design

Date: 2026-07-28
Status: approved design, pending written-spec review

## 1. Objective

Deliver a public, copy-and-run Kaggle notebook for standard CayleyPy permutation-puzzle competitions on exactly two NVIDIA T4 GPUs. A user attaches the standard `puzzle_info.json`, `test.csv`, and `sample_submission.csv`, supplies one supported MLP checkpoint, chooses a puzzle range and search modes, and runs the existing MultiGPUBeamSearch solver.

Validated results are also submitted best-effort to a public Cloudflare ingestion service. The service must safely accept at least 100 concurrent notebook publishers, preserve every accepted raw payload only until a confirmed GitHub commit, validate solutions, retain authorship and run provenance in GitHub, and publish append-only result records to `TryDotAtwo/cayleypy-beam-results` without exposing a GitHub write token.

The work is split into two independently testable deliverables:

1. Public Kaggle notebook plus repository helper CLI.
2. Cloudflare Results Ingest plus GitHub App publisher and repository CI validation.

The notebook remains useful when the ingest service or GitHub is unavailable.

## 2. Scope and Compatibility

### 2.1 Supported puzzle contract

The notebook supports the standard CayleyPy permutation-puzzle structure only:

- `puzzle_info.json` contains `central_state` and named permutation generators.
- `test.csv` contains puzzle ids and initial states in the standard CayleyPy representation.
- `sample_submission.csv` contains the corresponding puzzle ids and a path column.

Puzzle definitions, state length, class count, move count, and puzzle ids may differ between competitions. Non-permutation problems such as Sudoku or chess are out of scope.

The user selects an inclusive `PUZZLE_ID_START..PUZZLE_ID_END`. Every id in the interval must exist. Missing or duplicated ids fail preflight instead of being skipped.

### 2.2 Supported checkpoint contract

The public notebook exposes only `MODEL_SOURCE = "checkpoint"`. It automatically detects exactly one of:

- PilgrimAttnRes-style BatchNorm MLP, exported through the existing `batchnorm-folded` path.
- ResMLPDistance-style LayerNorm MLP, exported through the existing `resmlp-layernorm` path.

Detection uses checkpoint tensor keys and shapes. Ambiguous, mixed, incomplete, or unknown schemas fail before compilation. On T4, export dtype is always inferred as `fp16`; there is no public dtype setting.

The exported manifest must match the selected puzzle's state length and class count. Its head must be exactly `output_dim=1` or `output_dim=move_count`. Arbitrary PyTorch architectures and other output dimensions are explicitly unsupported.

### 2.3 Non-goals

- Automatic support for arbitrary PyTorch networks.
- Automatic adaptation of non-CayleyPy competition schemas.
- Publishing checkpoint weights or Kaggle secrets.
- Changing Stream2/3/4/5 algorithms merely to package the notebook.
- Claiming that Megaminx-measured profiles are optimal for every move count. They are safe measured T4 seeds with dynamic capacity guards.

## 3. Public Notebook User Contract

The first code cell is the only normal user-editable cell:

```python
AUTHOR_NAME
CHECKPOINT_PATH
PUZZLE_INFO_JSON
TEST_CSV
SAMPLE_SUBMISSION_CSV
PUZZLE_ID_START
PUZZLE_ID_END
BEAM_WIDTH
MAX_DEPTH
REFLECT_MODE                 # off | after_original | only
REFLECT_SOURCE_CSV           # required only for only
SOLUTION_MODE                # first | collect
COLLECT_UNTIL_DEPTH
MAX_COLLECTED_SOLUTIONS
TOUCH_BFS_RADIUS
PUBLISH_RESULTS              # true | false
RESULTS_INGEST_URL
```

Invalid enum values fail closed. `COLLECT_UNTIL_DEPTH` must not exceed `MAX_DEPTH`. `MAX_COLLECTED_SOLUTIONS` must be positive in collection mode. `REFLECT_SOURCE_CSV` must provide a valid original-oriented solution for every selected puzzle in `only` mode.

The notebook pins a solver git revision, clones under `/tmp`, writes durable logs and artifacts under `/kaggle/working`, and explicitly launches two ranks with torchrun. It validates exactly two T4 GPUs rather than inferring a generic topology from visible devices.

## 4. Notebook Architecture and Data Flow

The notebook is thin. Stable logic lives in small repository helpers with explicit CLI/data contracts:

1. `cayleypy_public_config.py`: parse and validate user configuration.
2. `cayleypy_data_adapter.py`: load and validate standard CayleyPy inputs.
3. `detect_export_mlp.py`: detect the supported checkpoint family and invoke the existing exporter.
4. `select_t4_profile.py`: choose a measured profile and derive safe dynamic capacities.
5. `run_cayleypy_2xt4.py`: orchestrate original/reflected and first/collect modes through the existing production runner.
6. `validate_cayleypy_solution.py`: replay paths independently on CPU.
7. `publish_results.py`: create bounded result envelopes and call the ingest endpoint best-effort.

Notebook cells perform:

1. Documentation and the user configuration.
2. Hardware, Internet, disk, schema, range, model, and memory preflight.
3. Repository checkout, checkpoint detection/export, profile selection, and build.
4. Two-rank execution with rank 0 streamed and full per-rank logs retained.
5. CPU validation, result aggregation, submission construction, and summary tables.
6. Best-effort publication and explicit publication status.

A helper failure includes the stage, puzzle id when applicable, safe error text, and relevant local artifact path.

## 5. Profile Selection and Capacity Safety

The requested beam is never replaced by a profile anchor. The selector:

1. Computes half-up rounded `log2(BEAM_WIDTH)`.
2. Clamps the profile key to `16..25`.
3. Chooses the measured table for `output1` or `output_move_count`.
4. Preserves requested beam and computes only the documented world/shard alignment.
5. Recomputes parent batching and Stream3/4/shard capacity guards from the actual move count and exported output dimension.

The notebook prints and saves requested beam, effective beam, alignment delta, selected anchor, evidence version, runtime parameters, and capacity derivation. For move counts other than the measured Megaminx 24, it labels the profile as a validated seed rather than claiming cross-puzzle optimality.

## 6. Search Modes

### 6.1 Touch BFS

`TOUCH_BFS_RADIUS` maps to `BEAM_SOLVED_NEIGHBORHOOD_RADIUS`. CPU builds the inverse-move neighborhood around the current target. GPU searches for a neighborhood touch. The stored suffix is appended during reconstruction and the complete path is replay-validated on CPU. Experimental K2/suffix expansion remains disabled and is not exposed publicly.

### 6.2 First-solution mode

`SOLUTION_MODE = "first"` stops after the first valid solution, validates it, and stores it as the candidate for submission.

### 6.3 Collection mode

`SOLUTION_MODE = "collect"` continues after target touches through `COLLECT_UNTIL_DEPTH`, subject to `MAX_COLLECTED_SOLUTIONS`. Every path is validated and deduplicated. Reaching the configured capacity is a successful bounded termination with `collection_status=capacity_reached`, not silent truncation.

The notebook writes `all_solutions.jsonl`, `solutions.csv`, and the best validated path per puzzle. Deduplication uses canonical path plus reached-state/provenance hashes so repeated distributed reports do not create duplicate records.

### 6.4 Reflection modes

- `off`: solve the original initial state only.
- `after_original`: solve original, construct reflected searches from validated original solutions, invert reflected results to original orientation, validate against the original state, and retain all provenance.
- `only`: read supplied valid original paths, build reflected searches without re-solving original, invert results, and validate against the original state.

The submission always receives the shortest valid original-oriented path found. Raw original and reflected records are retained separately.

## 7. Kaggle Artifacts

Each run writes:

- `selected_profile.json`
- `preflight.json`
- `run_summary.json`
- `beam_run_results.csv`
- `solutions.csv`
- `all_solutions.jsonl`
- `submission.csv`
- `publish_status.json`
- `beam_logs/` with combined and per-rank logs

Artifacts include schema version and a stable `run_id`. The notebook never embeds credentials, checkpoint tensors, private filesystem roots, or unbounded raw process environments.

## 8. Result Envelope

One result envelope is produced per validated solution. Required fields include:

- schema version, UUIDv7 submission id, client run id, and idempotency key;
- claimed author name, optional Kaggle username, and `author_verification=claimed`;
- Kaggle notebook owner/slug/version and public run URL when available;
- competition identifier, puzzle type, puzzle id, data/generator/state hashes, and a bounded canonical replay proof containing the initial state, central state, and named permutations required to validate the submitted path;
- original/reflected mode, reflected source id/path hash, and final orientation;
- solution path, length, solved/touch depth, validation result, and collection index/status;
- requested/effective beam, alignment delta, selected profile/evidence, and runtime config;
- touch BFS radius, solution mode, depth limits, and capacity limit;
- checkpoint filename, SHA-256, detected format, and sanitized exported manifest, never weights;
- GPU names, world size, rank/depth timing summaries, total runtime, solver commit, and timestamps;
- hashes and safe references for local/Kaggle artifacts.

Field and payload sizes are bounded. Unknown fields are rejected at the public API boundary until the schema version changes.

## 9. Cloudflare Results Ingest

### 9.1 Components

- Cloudflare Worker: HTTPS API, schema/size validation, rate limiting, idempotency lookup, and immediate receipt.
- R2: temporary immutable transport payload; source of recovery only until GitHub confirms the record commit, then deleted.
- D1: temporary submission state machine and retry metadata; the successful GitHub writer deletes the row after deleting its R2 object.
- Cloudflare Queue: at-least-once validation and publication work.
- Dead-letter queue: exhausted work retained for operator replay.
- Durable Object GitHub Writer: serializes repository mutations and batches validated records.
- GitHub App: repository-scoped write credential stored only as Cloudflare secrets.
- GitHub Actions: independent schema/path replay validation and staging-to-main merge gate.

### 9.2 Operating modes and state machine

`INGEST_MODE` is a case-sensitive exact allowlist with only `normal`, `store_only`, and `reject`. Missing, empty, mixed-case, or otherwise unknown values resolve to `reject`; raw configuration values are never returned or logged. In `normal`, the HTTP producer validates the envelope, persists immutable raw R2 plus a D1 row, and sends `{submission_id}` to the Queue. In `store_only`, it validates and persists raw R2 plus a D1 row that remains `received`, but performs zero Queue sends. In `reject`, including the fail-closed missing/unknown cases, the endpoint accepts nothing and performs no R2, D1, or Queue write.

In `normal`, the producer persists `received`, sends `{submission_id}` to the at-least-once Queue, and confirms the send with `received|queued|retryable -> queued`; the `queued -> queued` case is required for scheduled resend confirmation. Queue delivery may run before that confirmation, so the consumer must compare-transition `received|queued|retryable -> validating`. A successful confirmation refreshes `updated_at` and clears `ingest_paused` or any other stale `safe_error`. An ambiguous Queue failure returns `queued` only when a reread proves consumer progress beyond `queued`; otherwise an eligible `received|queued|retryable` row is confirmed `retryable`.

The downstream path is `validating -> validated | rejected -> staged -> published`; exhausted work becomes `dead_letter`. Retryable failures use bounded exponential backoff. Every consumer action and state transition is idempotent. A failed enqueue attempt increments `retry_count`, refreshes `updated_at`, and records only a safe error; a later successful enqueue clears that stale error. Raw R2 storage occurs before the Worker returns an accepted receipt.

### 9.3 Concurrency and loss prevention

The service target is at least 100 concurrent notebook clients. HTTP handlers do not call GitHub synchronously. In `normal`, they persist to R2/D1 and enqueue work, then return `202` with submission ids; in `store_only`, they persist the accepted receipt without enqueueing, as specified above. Queue consumers scale independently.

Every result uses a unique append-only repository path derived from schema version, competition, puzzle type, date, and submission id. No client-selected path is trusted. The Durable Object batches unique records into the staging branch, preventing competing GitHub ref updates. Index files are derived later and are never mutated by individual clients.

A normal concurrent duplicate waits a bounded interval for the first producer and reuses its receipt without another Queue send. If the shared row is still `received` after the final reread, the duplicate resends the same `{submission_id}` and confirms `queued`/later or `retryable`; this recovery may create another Queue delivery, so downstream consumer idempotency is mandatory. All such deliveries still reference one D1 row and one retained winning raw object.

The Worker scheduled handler is a strict no-op unless the resolved mode is `normal`. In `normal` it runs a bounded recovery page over stale `received|queued|retryable` rows, using the `(state,updated_at)` index and a `staleBefore` cutoff as backoff eligibility. It resends the same `{submission_id}`, uses checked state transitions, retains raw R2, and advances failed retry metadata so one hot page cannot starve older tail rows. A successful stale `queued` resend performs `queued -> queued`, refreshes `updated_at`, and clears `ingest_paused` or another stale `safe_error`, so the same row cannot flood every cron page. Thus a `store_only` row remains `received` even after it is stale; switching to `normal` lets the next scheduled run recover that same id once per eligible scan.

Queue backlog delivery resolves `INGEST_MODE` before inspecting the message body. For every non-`normal` mode, including missing/unknown values, it parses only `submission_id` and conditionally parks D1 `received|queued|retryable -> retryable` with `safe_error=ingest_paused`, unchanged `retry_count`, refreshed `updated_at`, and the raw R2 reference untouched. It rereads and verifies that durable park, then ACKs; a successful park never calls `message.retry()` and therefore consumes no Cloudflare `max_retries` attempt. Before that ACK it performs no R2 read, replay, publisher enqueue, token/GitHub access, payload log, or raw-mode log. A duplicate delivery repeats the same idempotent D1 park and still leaves one row. A D1 exception or unverifiable transition is not ACKed and follows the platform retry/exhaustion path; the unchanged stale row remains recoverable. When mode returns to `normal`, scheduled recovery resends the same parked `retryable` id, and it also rescues a stale `queued` row left after park failures exhausted Queue delivery.

At-least-once delivery cannot duplicate GitHub records because repository paths use the deterministic submission id and normalized GitHub data retains the idempotency key. R2, D1, Queue/DLQ, and Durable Object state are transient recovery mechanisms: they are retained during GitHub, DNS, Queue, or consumer outages, but the writer deletes the R2 object, D1 row, and pending Durable Object key after a confirmed GitHub commit. GitHub is the only long-term store.

### 9.4 Abuse controls

Because no user token is required, author identity is claimed, not authenticated. The public endpoint applies:

- strict body, field, path length, and batch count limits;
- per-IP and global rate limits;
- schema version allowlist;
- exact permutation replay validation before GitHub staging;
- duplicate and replay suppression;
- no arbitrary filenames, repository paths, markdown, workflows, or executable content;
- sanitized text fields and no secret echoing;
- the exact fail-closed `normal|store_only|reject` operating-mode gate described above.

Future OAuth can upgrade author verification without changing stored result identity.

## 10. GitHub Publication Model

Validated result records are written append-only to a staging branch in `TryDotAtwo/cayleypy-beam-results`. A periodic GitHub App branch/PR contains a bounded batch. CI:

1. validates JSON schema and file placement;
2. replays solutions using the included canonical puzzle proof bundle/hashes;
3. checks append-only policy and uniqueness;
4. regenerates deterministic indexes and summaries;
5. auto-merges only when all checks pass.

The Worker never holds a personal access token. GitHub App private key and installation id live only in Cloudflare Secrets. A publication failure never changes the notebook solve result; `publish_status=failed` includes a safe reason and submission ids for later status lookup.

The GitHub writer first persists accepted validated ids durably in Durable Object storage, then re-resolves `INGEST_MODE` immediately before any external token or GitHub API/ref/tree/commit write. If the mode is not `normal`, it makes no external request, does not mark rows `staged` or `published`, retains those durable ids, and re-arms a bounded alarm without logging payloads or the raw mode. The `setAlarm` call is part of the durable contract: if it fails, the operation throws and is retried rather than silently stranding ids. This final guard covers mode changes after Queue validation but before publication.

## 11. Testing and Acceptance

### 11.1 Notebook/helper tests

- config enum, inclusive range, and missing-id failures;
- checkpoint detection fixtures for both formats plus ambiguous/unknown failures;
- output dimension and manifest compatibility gates;
- profile selection anchors, half-up boundaries, clamping, exact beam preservation, alignment, and dynamic move-count capacity;
- original, reflection, collection, deduplication, capacity termination, and CPU path replay fixtures;
- result envelope schema, size limits, sanitization, and secret exclusion;
- notebook JSON/IPython-transform/AST validation;
- real Kaggle 2xT4 smoke and one bounded end-to-end public-version validation.

Existing real acceptance evidence for puzzle 0 at beam `2**21`, depth 100 remains recorded: output 1 solved with a validated length-60 path; output 24 solved with a validated length-57 path.

### 11.2 Ingest tests

- Worker schema/size/rate/idempotency tests;
- exact mode tests for `normal`, `store_only`, `reject`, missing, mixed-case, and unknown values, including zero persistence in fail-closed modes;
- scheduled-mode tests proving a `store_only` row older than 60 seconds is not enqueued, stale `queued` joins `received|retryable` only in `normal`, and a successful same-id resend refreshes `updated_at` and clears `ingest_paused`/stale errors before the next bounded page;
- property tests for permutation replay and malformed moves;
- D1 state transition and retry tests;
- Queue duplicate/out-of-order delivery tests;
- non-`normal` Queue backlog tests for `received|queued|retryable` proving durable `ingest_paused` parking, unchanged retry count/raw, reread verification, successful ACK with no message retry or prohibited binding access, idempotent duplicates beyond the configured `max_retries` equivalent, transition-false reread, a same-id normal resume, and D1 park exceptions through platform exhaustion followed by stale-queued recovery;
- R2-before-receipt durability test;
- Durable Object batching and simulated GitHub ref conflict tests;
- GitHub writer final-guard tests proving ids are durable before the mode guard, non-`normal` modes make zero external requests and re-arm an alarm, `setAlarm` failure throws/retries, and `normal` resume publishes retained ids;
- GitHub App permission and secret-leak tests;
- CI append-only and deterministic-index tests;
- load test with at least 100 concurrent publishers, retries, duplicates, mixed puzzles, and injected GitHub outage;
- recovery test proving every accepted receipt ends published, rejected with reason, or retained retryable/DLQ with its raw R2 payload.

### 11.3 Public release gates

- No public notebook push until local notebook/helper tests pass.
- No production ingest URL until staging load, durability, validation, and secret scans pass.
- Public notebook header states exact model and puzzle constraints.
- Kaggle kernel is explicitly public only after its metadata and rendered notebook are reviewed for secrets/private paths.
- Cloudflare and GitHub production secrets are installed through secret stores, never files or notebook cells.

## 12. Delivery Order

1. Implement and test repository helpers and the public notebook against a local/mock ingest endpoint.
2. Implement Cloudflare ingest storage, validation, queueing, and status API in staging.
3. Implement GitHub App staging publisher and results-repository CI.
4. Run 100-client staging load/recovery tests.
5. Deploy production ingest and verify a bounded end-to-end result.
6. Secret-scan and publish the Kaggle notebook publicly.
7. Monitor the first public submissions and retain an emergency publication-disable switch.

This order prevents the public notebook from depending on an unvalidated ingestion service while keeping solve artifacts useful independently.
