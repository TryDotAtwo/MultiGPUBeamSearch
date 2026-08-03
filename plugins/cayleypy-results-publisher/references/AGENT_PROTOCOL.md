# CayleyPy result submission protocol for agents

Use `../scripts/cayleypy_submit.py`; do not handcraft HTTP, gzip, identifiers, proof hashes, or endpoint URLs.

1. Select schema v1 only for Kaggle runs with the required `kaggle` object. Select v2 for SLURM, clusters, workstations, and other native runs.
2. Prefer a solver-produced complete JSON/JSONL envelope. Use CSV/TSV or `.moves` only with a completed publisher config containing all run metadata and puzzle proof context.
3. Never infer or fabricate author, competition, puzzle type, initial/central states, generators, orientation/reflection, model identity/hash/format, hardware, beam/profile, runtime, run id, solver commit, or timings.
4. Run `preflight INPUT [--config CONFIG]`. Stop on any nonzero exit.
5. Run `submit INPUT [--config CONFIG] --manifest cayleypy-receipts.json --wait`. The endpoint is deliberately pinned. Do not request a user token or endpoint.
6. Preserve the manifest and retry the identical command after a network failure. Accepted parts are skipped by compressed SHA-256.
7. Treat HTTP 202 as accepted only. Claim GitHub publication only when the CLI reports zero rejected and unresolved receipts. A missing Worker status after D1 cleanup is verified against the expected immutable GitHub raw URL.
8. Report counts and manifest path. Do not print input envelopes, submitted archives, authorization values, or environment secrets.

Accepted input forms:

- one canonical envelope object;
- `{ "schema_version": 1|2, "results": [...] }`;
- JSONL with one envelope per line;
- CSV/TSV with the exact header in `../templates/solutions.csv`;
- `.txt` or `.moves` containing a dot-separated path, only when config has exactly one puzzle context.

The client derives UUIDv7 submission ids, UTC timestamps, proof hashes, semantic idempotency keys, deterministic canonical JSON and gzip, and partitions batches at 32 MiB compressed / 64 MiB raw. Preserve input order. Do not modify beam-search/CUDA code as part of result publication.