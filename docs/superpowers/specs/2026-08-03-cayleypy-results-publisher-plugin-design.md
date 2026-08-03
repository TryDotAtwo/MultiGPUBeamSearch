# CayleyPy Results Publisher Codex Plugin Design

## Goal

Provide a public Codex plugin and a standalone Python CLI that let any user publish replay-verifiable CayleyPy solutions from Kaggle, SLURM, or an ordinary machine through the existing anonymous Cloudflare ingest service into the append-only GitHub results repository.

The plugin must be useful to both a human following a recipe and an agent executing a deterministic protocol. It must not require GitHub or Cloudflare credentials and must not change the beam-search implementation.

## Chosen approach

Use a repository-hosted Codex plugin backed by a dependency-free Python CLI. Documentation-only guidance is too error-prone for canonical JSON, gzip boundaries, receipts, and polling. An MCP server would add an unnecessary service and would be awkward inside Kaggle and cluster jobs.

## Repository layout

The implementation lives on `codex/cayleypy-results-ingest`:

```text
plugins/cayleypy-results-publisher/
  .codex-plugin/plugin.json
  skills/publish-cayleypy-results/SKILL.md
  scripts/cayleypy_submit.py
  references/HUMAN_GUIDE_RU.md
  references/AGENT_PROTOCOL.md
  templates/kaggle-v1.json
  templates/native-v2.json
```

A repository marketplace entry makes the plugin discoverable and installable from `MultiGPUBeamSearch`. The manifest exposes only the skill; no MCP server, application, secret, or hook is required.

## Inputs and supported sources

The CLI accepts one canonical envelope object, a versioned batch object `{schema_version, results}`, or JSONL containing one canonical envelope per non-empty line.

It supports schema v1 for Kaggle-produced results sent to `/v1/results`, and schema v2 for native SLURM, other clusters, and ordinary machines with truthful native provenance sent to `/v2/results`. Version inference must be unambiguous. Mixed-version input is rejected locally. A bare move string is insufficient because publication requires puzzle, proof/replay, model, runtime, author, and provenance data. Templates and guides explain how producers fill those fields; the server remains the authoritative validator.

## CLI contract

Primary command:

```text
python cayleypy_submit.py submit PAYLOAD --endpoint-base URL --wait
```

Behavior:

1. Parse and normalize supported input containers without changing envelope contents.
2. Infer or verify one schema version and select `/v1/results` or `/v2/results`.
3. Perform bounded local structural checks and reject secrets, mixed schemas, redirects, and unsafe endpoint schemes.
4. Partition only between envelopes. Serialize each batch as canonical UTF-8 JSON and deterministic gzip.
5. Keep every compressed request at or below 32 MiB and every decompressed request at or below 64 MiB. A single oversized envelope fails locally.
6. Send parts sequentially with archive index/count headers. Never follow redirects.
7. Accept only safe JSON responses. Persist a receipt manifest atomically after each accepted part so interruption can resume without losing evidence.
8. With `--wait`, poll the returned same-origin `/v1/submissions/<id>` URLs until every receipt is terminal or a bounded timeout is reached.
9. Exit non-zero on validation, transport, partial acceptance, rejected terminal state, or timeout. Duplicate/idempotent receipts are successful.

Default output is concise and safe: counts, part numbers, submission IDs, idempotency keys, statuses, and file paths. It never prints submitted envelopes, authorization headers, environment secrets, or full server internals.

## Human workflow

The Russian human guide provides copy-paste recipes for Kaggle publication of either the best or every collected solution, native/SLURM jobs, local workstations, receipt resume/polling, and GitHub verification. It distinguishes HTTP 202 from completed GitHub publication and explains claimed authorship, idempotency, size limits, and canonical metadata.

## Agent protocol

The agent must discover the producer artifact and schema version, validate truthful provenance, refuse fabricated fields, run a local dry-run, submit bounded archives sequentially, save receipts, poll when requested, and report exact accepted, duplicate, published, rejected, and unresolved counts. It must never claim GitHub publication from HTTP 202 alone. The skill routes the agent to the CLI instead of recreating HTTP logic ad hoc.

## Error handling and resilience

- Deterministic idempotency makes retry after timeout safe.
- Network failures use bounded exponential backoff with `Retry-After` support.
- A partially accepted multi-part run records completed receipts and stops with a resumable manifest.
- Redirects, cross-origin status URLs, malformed JSON, unknown statuses, oversized responses, and mixed versions fail closed.
- Cloudflare remains transient transport/storage; GitHub is the long-term source of truth.

## Testing and acceptance

Tests use only the Python standard library and a local mock HTTP server. They cover parsing, routing, deterministic gzip, size boundaries, oversized records, archive headers, redirects, retry, partial acceptance, resume, same-origin polling, safe logs, and exit codes.

Acceptance requires plugin and skill validation, passing CLI tests on Windows and Linux-compatible Python, green existing Worker tests/typecheck, dry-runs against checked-in v1/v2 examples, a secret-free staging receipt/publish smoke, and updated project memory/test evidence.

## Non-goals

- Changing beam-search or CUDA code.
- Guessing missing proof/provenance from a move string.
- Hosting a second server or MCP service.
- Storing durable result archives outside GitHub.
- Embedding public write tokens or private credentials.
