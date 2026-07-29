# Public CayleyPy Task 6 canonical reconciliation verification

Date: 2026-07-29
Base: `c53f52dfd18189ebf8ad75123abaa1962ee910db`

## Scope

- Replaced the legacy producer-only result envelope with the canonical replay-capable v1 batch schema.
- Added three shared exact-byte/hash fixtures: original plus UTF-8 author and slash puzzle type, reflected plus source provenance, and empty source path.
- Reconciled producer sanitization, semantic idempotency, UUIDv7/UTC transport fields, proof hash recomputation, original-oriented token arrays, reflection source provenance, full profile/runtime/model/hardware fields, and integer microsecond timings.
- Exported the strict path tokenizer and retained raw searched/reflection-source paths through runner records.
- Added public input preflight for the fixed `1 <= state_len <= 120` State128 logical payload.
- Kept the 256 KiB envelope, 100-result, and 4 MiB request bounds.
- Did not contact an external service or edit the ingest worktree, checkpoint export algorithms, profiles, beam/CUDA/C++ algorithms, or Stream 1-5.

## RED/GREEN record

- Missing shared fixture: `1 failed, 14 deselected`; fixture then exposed old envelope schema: `1 failed, 14 deselected`; canonical schema: `1 passed, 14 deselected`.
- Old producer against three goldens: `3 failed, 15 deselected`; canonical producer: all golden cases passed.
- Model head semantics: `2 failed, 18 deselected`; guards added and `2 passed`.
- Publisher model-head semantics: inconsistent but schema-valid/rehashed envelope reached the fake HTTP boundary as retryable; semantic publish validation added and the case is now rejected locally/non-retryably.
- Slash puzzle type: `1 failed, 19 deselected`; dedicated puzzle-type grammar added and `1 passed`.
- Oversized public state: `1 failed, 4 deselected`; loader preflight added and all `5 passed`.
- Canonical schema state cap: `1 failed, 19 deselected`; max logical state length changed to 120 and `1 passed`.

## Canonical fixture checks

Node and Python independently reproduced all three stored canonical JSON byte strings, full-envelope SHA-256 values, and semantic SHA-256 values. Both observed the UTF-8 author code points:

`1040,1083,1080,1089,1072,32,916`

No code point was 63 (`?`), so the fixture exercises real non-ASCII canonicalization.

## Final gates

- `python -m pytest -q tests/cayleypy_public/test_results.py` -> `21 passed`.
- `python -m pytest -q tests/cayleypy_public/test_data.py tests/cayleypy_public/test_paths.py tests/cayleypy_public/test_runner.py tests/cayleypy_public/test_results.py` -> `102 passed`.
- `python -m pytest -q tests/cayleypy_public` -> `141 passed`.
- `python -m pytest -q` -> `174 passed`.
- Node canonical verifier -> `3 passed`.
- Python canonical verifier -> `3 passed`.
- Draft 2020-12 meta-validation and fixture validation -> passed.
- Python compile -> passed.
- Public schema/golden forbidden-field, sentinel-secret, absolute-private-path, tensor-payload, and collapsed-Unicode scan -> no hits. Source/tests were separately inspected; sensitive-looking strings there are synthetic negative-test sentinels and allowlist field names, not serialized public data or real credentials.
- `git diff --check` -> passed.
- Exact staged payload inspection -> only the canonical-contract, provenance, preflight, tests, plans/spec, report/ledger, and memory files listed in the commit.
