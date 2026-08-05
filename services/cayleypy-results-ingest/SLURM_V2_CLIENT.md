# CayleyPy native SLURM results API v2

`POST https://cayleypy-results-ingest-staging.tupa-expert.workers.dev/v2/results`
accepts only schema v2 native-cluster records. `POST /v1/results` remains the
unchanged Kaggle API and rejects v2.

## Client contract

1. Build an envelope conforming exactly to
   `configs/cayleypy_results_schema_v2.json`; unknown fields are rejected.
2. Set `author.verification` to `claimed`. `cluster_name` is also claimed
   provenance, not verified identity.
3. Report the real SLURM job, release manifest, GPU count, native SM, VRAM,
   driver and measured/bounded profile. Never synthesize Kaggle fields.
4. Compute `idempotency_key` as SHA-256 of canonical JSON after removing only
   `client_submission_id`, `run_id`, `idempotency_key`, and `submitted_at`.
   Canonical JSON sorts every object key recursively, uses compact separators,
   preserves array order, and encodes UTF-8.
5. Send `{ "schema_version": 2, "results": [...] }` as `application/json`, or
   deterministic gzip as `application/gzip`. A gzip request is at most 32 MiB
   compressed and 64 MiB after decompression; split larger result sets at
   envelope boundaries and send parts sequentially.
6. A successful durable handoff returns HTTP 202 with `receipts[]`. Poll the
   returned `status_url`; the shared status contract intentionally remains
   `GET /v1/submissions/<submission_id>`.
7. Do not follow redirects. A response/error never contains the submitted
   payload, secrets, raw ingest mode or GitHub internals.

Canonical examples:

- `configs/cayleypy_results_v2_golden.json` contains original and reflected
  replay-valid envelopes.
- `configs/cayleypy_results_v2_example_payload.json` is a directly POSTable
  H100x4 original batch.

Publication is append-only under
`data/v2/slurm/<competition>/<puzzle_type>/<yyyy-mm-dd>/<submission_id>.json`.
The record preserves `provenance.platform=slurm` and claimed author data, and is
never mixed with v1 Kaggle paths.

## Minimal curl

```bash
curl --fail-with-body --max-redirs 0 \
  -H 'Content-Type: application/json' \
  --data-binary @configs/cayleypy_results_v2_example_payload.json \
  https://cayleypy-results-ingest-staging.tupa-expert.workers.dev/v2/results
```

No D1 migration is needed: v2 uses the existing generic receipt/state machine,
while the immutable raw object key (`raw/v2/...`) and GitHub path carry the
schema namespace. Cloudflare storage is transient until GitHub commit; GitHub
remains the only long-term store.