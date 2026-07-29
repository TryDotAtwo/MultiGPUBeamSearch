# Task 6 result envelope and best-effort publisher report

Date: 2026-07-29
Base commit: `4305300118fb464e6a31d069f1185010e3a02bb4`
Review-fix base commit: `93d9394bd15f4168325d18339b07516db484abcb`

## Outcome

Implemented the Task 6 contract without integrating it into the solver or performing any external publish. The result layer is isolated to `configs/cayleypy_results_schema_v1.json`, `tools/cayleypy_public/results.py`, and `tests/cayleypy_public/test_results.py`.

## Envelope contract

- Draft 2020-12 schema version 1 requires author, Kaggle kernel slug/version/notebook hash, replay proof bundle, solution, measured profile/runtime, checkpoint SHA-256 plus sanitized exporter manifest, hardware, timings, and solver commit.
- Every object is fail-closed with `additionalProperties: false`. The builder copies only explicit allowlists, so unknown fields, environment mappings, tokens, absolute checkpoint paths, `source_weights`, and tensors cannot enter the envelope.
- Canonical JSON uses sorted keys, compact separators, UTF-8, and rejects non-finite or non-JSON values.
- Submission ids are UUIDv7. The SHA-256 idempotency key covers the complete sanitized semantic payload but excludes the unique submission id itself.
- Both build and publish paths enforce the exact schema and the 256 KiB per-envelope limit. The publisher recomputes the semantic idempotency key before HTTP.
- The manifest contract uses the real exporter field `nrd`; no model, profile, runner, beam, or CUDA code changed.

## Publisher contract

- Requests are canonical `POST` bodies shaped as `{"schema_version":1,"results":[...]}`, limited to 100 envelopes, and rejected locally as non-retryable when the complete canonical body exceeds 4 MiB.
- HTTP 202 is accepted; HTTP 200 is a duplicate success. HTTP 429 and 5xx plus timeout/DNS failures are retryable failures. Schema, endpoint, timeout, empty-request, and request-size failures are non-retryable.
- Publishing is best-effort: ordinary validation, HTTP, DNS, timeout, and persistence exceptions are converted to `PublishStatus` instead of escaping into solve completion.
- `publish_status.json` is atomically written and fsynced before normal return. It stores only endpoint scheme plus host and optional port, never userinfo, path, query, or fragment, together with a bounded safe error.
- Response headers and bodies are never copied or read into status. Safe errors are generic, sanitized, and at most 2 KiB.

## TDD evidence

- Envelope RED: four assertions failed because the builder/schema were absent (`4 failed`). GREEN: schema, provenance, UUIDv7/idempotency, redaction, and size tests passed (`4 passed`).
- Publisher RED: eight assertions failed because the publisher was absent (`8 failed, 4 deselected`). GREEN: 202, duplicate 200, timeout, DNS, 429, 500, schema-before-HTTP, and 100-result-limit cases passed (`8 passed, 4 deselected`).
- Exporter compatibility RED: the realistic `nrd` manifest fixture failed because the first schema draft required `blocks`. GREEN changed the schema/allowlist to `nrd` and restored all 12 focused tests.
- Review RED: the path-secret status test retained `/private/path-secret` (`1 failed, 13 deselected`); the request-bound pair let a 100-envelope near-limit body cross the HTTP boundary (`1 failed, 1 passed, 12 deselected`).
- Review GREEN: origin-only status plus 100-small accepted and 100-near-limit locally rejected (`3 passed, 11 deselected`), then all focused tests passed.

## Verification

- `py -3 -m pytest -q tests/cayleypy_public/test_results.py` -> `14 passed`.
- `py -3 -m pytest -q tests/cayleypy_public` -> `119 passed`.
- `py -3 -m pytest -q` -> `152 passed`.
- Durable verification note: `test_results/cayleypy_public_task6_2026-07-29.md`.
- JSON Schema meta-validation, public-artifact secret scan, and `git diff --check` are final commit gates.

No external endpoint was contacted, no result was deployed or published, and no package, runner, model, beam, CUDA, or ingest implementation was changed.
