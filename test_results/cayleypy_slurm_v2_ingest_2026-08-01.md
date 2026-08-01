# Native SLURM results ingest v2 verification — 2026-08-01

## Scope

Implemented a backwards-compatible `POST /v2/results` in the existing Cloudflare Worker. `POST /v1/results` remains the Kaggle-only API. No beam-search/CUDA code, production deployment, Cloudflare secret, or D1 schema was changed.

Starting point: branch `codex/cayleypy-results-ingest`, commit `3df4d9effdecc787e6acc632b36b4eefd63080c9`.

## Contract

- Strict schema v2 rejects unknown/private fields and accepts only native SLURM provenance.
- Hardware is 1..16 real ranks with matching GPU-name cardinality, world size, native SM, VRAM and hardware-bound profile.
- Profile status is only `measured` or `bounded_from_measured`; backend/model, beam alignment and profile anchor are cross-checked.
- Canonical replay, proof and orientation validation is reused from v1; v2 modes are `off`, `after`, and `only`.
- Durable raw retry keys are versioned as `raw/v2/...`.
- Published records are isolated at `data/v2/slurm/<competition>/<puzzle_type>/<yyyy-mm-dd>/<submission_id>.json`.
- Shared status receipts intentionally continue to use `/v1/submissions/<submission_id>`.
- No D1 migration is required because existing state rows are version-neutral and immutable raw/GitHub keys carry the version.

## TDD evidence

- Schema RED: v2 tests failed because `schema-v2` did not exist; existing 57 v1 schema tests remained green.
- Route RED: `/v2/results` returned 404 before route implementation.
- Queue RED: v2 raw objects did not enqueue the GitHub writer before version dispatch.
- Writer RED: `resultPathV2` was absent before the separate namespace was implemented.
- GREEN: 72 schema/config tests and 138 Worker/Queue/GitHub tests passed.
- TypeScript: `tsc --noEmit` passed.
- Mixed concurrency: 100 simultaneous v1/v2 publishers all returned HTTP 202 and created 100 durable D1 rows.
- Extra v2 coverage: original/reflected golden replay, directly POSTable example, gzip, idempotent duplicate POST/Queue delivery, hardware matrix, invalid rank/cardinality/mixed hardware/cross-profile/backend, mode failures, GitHub path conflict and v1/v2 route isolation.

## Public artifacts

- `configs/cayleypy_results_schema_v2.json`
- `configs/cayleypy_results_v2_golden.json`
- `configs/cayleypy_results_v2_example_payload.json`
- `services/cayleypy-results-ingest/SLURM_V2_CLIENT.md`

## Live staging evidence

- Implementation commit: `765446a`.
- GitHub/Cloudflare Workers Build `2c649d08-7649-4976-933c-c3a8339cea2b`: `completed / success` for `cayleypy-results-ingest-staging`.
- Direct Windows clients could not complete the local TLS handshake, so a private, CPU-only Kaggle network probe sent the exact checked-in example without secrets. Smoke notebook v4 completed and received HTTP 202.
- Receipt submission: `019fbdc0-9347-7639-895c-d27e703694ad`; idempotency: `76e18ada78a88fd4e94fed92a80896ff9564409bc9439b7d7f20c6599a260fde`.
- GitHub `ingest/staging` contains `data/v2/slurm/toy-cayley/cube_3-3-3/2026-07-29/019fbdc0-9347-7639-895c-d27e703694ad.json` (blob `3583efd574ad3a41abc9c5fc936ec2e1cbf0dc6c`). The fetched record has `schema_version=2`, `provenance.platform=slurm`, `world_size=4`, and `native_sm=90`.
- Production deployment remains explicitly out of scope and was not performed.
