# Public CayleyPy Task 6 verification

Date: 2026-07-29

## Scope

- Added the exact Draft 2020-12 result schema, isolated result-envelope/publisher module, and focused tests.
- Did not integrate publishing into solve orchestration and did not contact an external endpoint.
- Did not change package, runner, model, beam, profile, CUDA, or ingest code.

## TDD record

- Envelope/schema RED: `4 failed` because the builder/schema were missing.
- Envelope/schema GREEN: `4 passed`.
- HTTP/status RED: `8 failed, 4 deselected` because the publisher was missing.
- Exporter-shape RED: realistic `nrd` manifest failed the initial `blocks` schema.
- Combined GREEN: `12 passed`.

## Final gates

- `py -3 -m pytest -q tests/cayleypy_public/test_results.py` -> `12 passed`.
- `py -3 -m pytest -q` -> `150 passed`.
- Python compile and Draft 2020-12 schema meta-validation passed.
- Public source/artifact secret scan and diff checks passed.

The publisher tests cover canonical 202 POST, duplicate 200, timeout, DNS, 429, 500, invalid schema before HTTP, the 100-result request limit, atomic local status, credential redaction, no response-body/header capture, and safe errors bounded to 2 KiB.
