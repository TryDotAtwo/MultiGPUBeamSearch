# Task 6 canonical result-contract reconciliation report

Date: 2026-07-29
Base commit: `c53f52dfd18189ebf8ad75123abaa1962ee910db`

## Outcome

Reconciled the public producer with one replay-capable canonical v1 batch contract. The previous producer-only envelope and the earlier ingest draft disagreed on identity, proof, reflection, collection, profile, runtime, model, hardware, and timing fields. This slice replaces the public schema and producer contract, preserves runner provenance needed by the canonical envelope, and adds shared byte-exact fixtures for producer/ingest parity.

No external endpoint was contacted. The ingest worktree was read only and was not edited. No checkpoint export algorithm, measured profile, beam-search algorithm, CUDA/C++, Stream 1-5, or GPU execution path changed.

## Canonical v1 batch contract

- The root is exactly `{"schema_version":1,"results":[...]}` with 1-100 closed result objects.
- Transport identity is `client_submission_id` UUIDv7 plus `run_id`, UTC `submitted_at`, and SHA-256 `idempotency_key`.
- Semantic idempotency excludes exactly `client_submission_id`, `idempotency_key`, `submitted_at`, and `run_id`; every other canonical field participates.
- Proofs retain full logical `initial_state`, `central_state`, and generator permutation arrays, plus producer-recomputed canonical SHA-256 hashes for those values and the reached state.
- Logical state arrays are bounded to 1-120 entries, matching `State128.v[0..119]`; the public input loader rejects larger states before export, build, or CUDA launch.
- Standard slash-bearing puzzle types such as `cube_3/3/3` use a dedicated bounded grammar. Competition and run identifiers retain their separate path-safe grammar.
- Solutions use strict printable-ASCII, dot-free move tokens and store the CPU-validated original-oriented path as an array. Reflected results also retain the exact searched path, source path, and SHA-256 of the exact source dot string.
- Full selected profile, runtime tuning, model filename/hash/format/sanitized manifest, hardware, and integer-microsecond timings are retained. Unknown, private, tensor, environment, token, and absolute-path inputs are omitted by allowlist.
- Producer semantic checks require `output1` manifests to have `output_dim=1` and `output_move_count` manifests to have `output_dim` equal to the generator count.
- The existing 256 KiB per-envelope, 100 results per request, and 4 MiB canonical request limits remain enforced before HTTP.

## Shared fixtures and canonicalization

`configs/cayleypy_results_v1_golden.json` contains three cases in stable order:

1. Original orientation with a real UTF-8 author (`Алиса Δ`) and standard-style `cube_3/3/3` puzzle type.
2. Reflected orientation with searched path, exact reflection source path, and source hash.
3. Source orientation with an empty solution path.

Each case stores the exact canonical UTF-8 JSON string, full-envelope SHA-256, and semantic SHA-256. Independent Node and Python canonicalizers reproduced all three byte strings and hashes. The author code points were verified as `1040,1083,1080,1089,1072,32,916`, with no replacement `?` code point.

## TDD evidence

- Shared-contract RED: the golden fixture was absent, then the old envelope-only schema rejected the batch contract. GREEN: schema meta-validation and all three fixtures passed.
- Producer RED: all three golden cases failed because the old builder required `proof_bundle`. GREEN: canonical producer bytes and hashes matched all three fixtures.
- Model semantic RED: two invalid model/profile combinations were accepted (`2 failed`). GREEN: both are rejected before schema publication.
- Publisher semantic RED: a schema-valid envelope with a recomputed idempotency key and inconsistent output head reached the HTTP boundary as retryable. GREEN: publish validation rejects it locally as a non-retryable payload error.
- Standard puzzle-type RED: `cube_3/3/3` still referenced the slash-forbidding generic identifier (`1 failed`). GREEN: the dedicated puzzle-type definition validates the golden.
- State-cap RED: a 121-element puzzle reached contract loading (`1 failed`). GREEN: public preflight rejects it with the fixed `State128` logical-capacity message.
- Schema-cap RED: the result schema still allowed 128 logical state entries (`1 failed`). GREEN: the canonical proof-state bound is 120.

## Verification

- `python -m pytest -q tests/cayleypy_public/test_results.py` -> `21 passed`.
- `python -m pytest -q tests/cayleypy_public/test_data.py tests/cayleypy_public/test_paths.py tests/cayleypy_public/test_runner.py tests/cayleypy_public/test_results.py` -> `102 passed`.
- `python -m pytest -q tests/cayleypy_public` -> `141 passed`.
- `python -m pytest -q` -> `174 passed`.
- Independent Node and Python canonical fixture checks -> `3 passed` in each runtime with identical UTF-8 code points and SHA-256 values.
- Python compile, Draft 2020-12 schema meta-validation, secret/private-path/tensor scans, exact staged-payload inspection, and whitespace checks are final commit gates.

The future Task 7 notebook contract now explicitly requires its header to place the fixed `1 <= state_len <= 120` limit beside both supported checkpoint families and the `output_dim=1` or exact-`move_count` head rule.
