# CayleyPy results ingest Task 8 verification (2026-07-30)

## Outcome

Task 8 adds a deterministic 100-publisher staging workload and a bounded
post-recovery audit without changing CUDA, C++, the solver, or beam-search
architecture.

The implementation is locally complete and passed the exact private Kaggle CPU
runtime gate. This gate validates the workload/audit contracts and the complete
Worker regression suite. It does not claim that a live Cloudflare staging load,
controlled GitHub outage, or production deployment has occurred.

## Deterministic workload

`services/cayleypy-results-ingest/load/k6-100-publishers.js` uses
`exec.scenario.iterationInTest` to map exactly 100 global iterations:

- `0..79`: 80 semantically unique valid results across 20 authors;
- `80..89`: 10 transport-only duplicates of `0..9`, retaining the exact
  semantic idempotency key;
- `90..99`: 10 invalid reached-state proof hashes, with idempotency recomputed
  over the deliberately invalid semantic payload.

The exact k6 profile is:

```javascript
export const options = {
  vus: 100,
  iterations: 100,
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<2000"],
    checks: ["rate>0.99"],
  },
};
```

Initial jitter is deterministic and below 200 ms. A `429` honors
`Retry-After`, capped at 60 seconds, adds deterministic jitter below 250 ms,
and retries at most three times. Exhausted `429`, malformed receipts,
unexpected statuses, and all `5xx` responses fail the iteration.

The workload accepts only `INGEST_BASE_URL` and the non-secret
`LOAD_PHASE=baseline|github-outage|recovery`. It has no token or authorization
input. Logs contain only bounded receipt identifiers:

```text
CAYLEYPY_RECEIPT<TAB>{"type":"receipt","workload_index":0,"case_kind":"valid","submission_id":"...","idempotency_key":"...","status_url":"..."}
```

It never logs request envelopes or response bodies.

## Bounded recovery audit

`services/cayleypy-results-ingest/test/recovery-audit.ts`:

- bounds the receipt manifest to 1 MiB and 200 receipt events;
- bounds D1, R2, and GitHub snapshots to 4 MiB and 2,000 rows;
- bounds status responses, poll intervals, and total poll time;
- deduplicates receipt events by submission id and verifies semantic duplicate
  convergence;
- fetches status only from the configured `INGEST_BASE_URL`, never the
  manifest-provided host;
- checks matching D1 rows, live status, retained immutable R2 objects, unique
  GitHub paths/idempotency keys, and one GitHub file per published key;
- in strict mode requires all 80 validated unique results to be published;
- in `--allow-recoverable` mode permits only terminal `rejected` or retained
  `retryable|dead_letter` evidence with a safe reason.

The audit accepts sanitized inventories only and has no Cloudflare or GitHub
token input.

## Exact private Kaggle CPU gate

Kernel:
`https://www.kaggle.com/code/trydotatwo/cayleypy-results-ingest-npm-gate`

Final version: **v35**.

Pulled metadata proves:

- `is_private=true`;
- `enable_gpu=false`;
- `enable_tpu=false`;
- `enable_internet=true`;
- kernel id `trydotatwo/cayleypy-results-ingest-npm-gate`.

Runtime:

- Node `v22.23.1`;
- npm `10.9.8`;
- `@cloudflare/vitest-pool-workers=0.19.0`;
- `@cloudflare/workers-types=5.20260729.1`;
- `vitest=4.1.10`;
- `wrangler=4.115.0`;
- resolved `miniflare=4.20260722.1`;
- sole `workerd=1.20260729.1`.

Results:

- kernel terminal status: `KernelWorkerStatus.COMPLETE`;
- all 14 gate commands exited zero;
- Task 8 static/unit tests: 12/12 passed in 70 ms;
- schema, Wrangler, and Task 8 pool: 28/28 passed in 742 ms;
- full real Worker/Miniflare pool: 119/119 passed in 20.39 s;
- independent Worker rerun: 119/119 passed in 18.31 s;
- TypeScript `tsc --noEmit`: passed;
- exact npm registry versions: passed;
- exact resolved stack and sole workerd override: passed;
- compatibility-warning scan for date `2026-07-28`: zero hits;
- gate assertions were emitted at 75.438 s and notebook output finalized at
  84.273 s.

## Provenance and exact hashes

Both the generated and Kaggle-pulled notebooks parse as JSON, and the
concatenated code cells parse as Python AST. Kaggle execution changes the full
notebook bytes by adding outputs, but the ordered markdown/code source cells
are byte-equivalent under canonical JSON:

- source-cell SHA-256, generated and pulled:
  `528cbf5b2f9f48e99220457f955ac820ca6ec5badab99e9bd92b5a7ab23f6a6f`;
- generated notebook SHA-256:
  `7d830c244b860edf7a58aa1d9fdabde4431f543475bd903768c739739940c188`;
- pulled executed notebook SHA-256:
  `93b821a1fbf3cb46491d6dc32848af3f6d6d11caa5278ea315a3ad356e1615fb`;
- pulled metadata SHA-256:
  `572d93ed4dd74d3aeabba1951ab4e098cbf6d217219c5b9edddab17bdbb35ae0`.

The notebook's ordered `EXPECTED_SHA256`, embedded zip members, runtime
`payload-sha256.json`, post-install manifest, `npm-gate-results.json`, and
current checkout all match for all **33/33** payload files. Post-install
mismatch count is zero.

Task 8 payload hashes:

- k6 workload:
  `754dfd154b554f5434815e5215c575758f605ed5285f4e374b7cc4fda45c3889`;
- Task 8 Vitest gate:
  `7176b9ff0c5d7e8dbc4ddc215627857c6b336c9e0eb4f5ed3d8e8fa2aac417ae`;
- recovery audit:
  `c834070f09a788d343b73bb6e6a345042912604eb9f80d84a545155d3b49f58a`.

Evidence hashes:

- `npm-gate-results.json`:
  `669cb8bb6f4b900ab39824cbb151aa8d24cc8fc7fc7d3e7bf2867b1559bdc521`;
- pre/post payload manifest (identical):
  `8a805d77934f65a2d94164b22a3c223345a02b5d51fc49d43c12e91f7d06630b`.

Downloaded outputs, logs, pulled source, and metadata are retained under
`test_results/kaggle_cayleypy_results_ingest_npm_gate/task8_v35/`.

## Remaining live acceptance gate

The following require an explicitly configured Cloudflare staging deployment
and controlled GitHub-outage window; they were not fabricated by the CPU gate:

1. baseline and outage k6 summaries plus safe receipt manifests;
2. Worker metrics, Queue backlog, D1 rows, R2 inventory, and DLQ count;
3. strict recovery audit proving 80 unique results published, 10 duplicates
   converged, 10 invalid proofs rejected before receipt, retained raw evidence,
   and exactly one repository file per idempotency key.
