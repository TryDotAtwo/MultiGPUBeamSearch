# CayleyPy Results Publisher plugin verification — 2026-08-03

## Scope

Built a public Codex plugin plus a dependency-free Python CLI on `codex/cayleypy-results-ingest`. The client accepts complete Kaggle v1/native v2 JSON or simple move/CSV/TSV input expanded from a fill-once config. It pins the existing anonymous Cloudflare staging endpoint, creates deterministic bounded gzip batches, persists safe receipts, resumes accepted parts, and verifies terminal publication without user tokens. Beam-search/CUDA code was not changed.

- Starting commit: `ce40a15f` (`Record live SLURM v2 staging evidence`).
- Implemented commits: `4b364f5b`, `2e4ac185`, `a1d4daba`, `157f761a`, `fcd91c5a`, `1c6bc0f4`.
- Verified HEAD before final evidence: `1c6bc0f40f55eeb18f1354082db554656435e3f1`.
- Runtime: Python 3.11.5, Node v22.22.0, npm 10.9.4.

## Public contract

- Official origin is compiled into the client: `https://cayleypy-results-ingest-staging.tupa-expert.workers.dev`.
- No GitHub or Cloudflare credential is accepted in the normal CLI.
- One deterministic gzip request per archive, with 32 MiB compressed and 64 MiB raw limits; oversize collections split sequentially only between envelopes.
- Supported inputs: complete envelope/batch JSON, JSONL, exact-contract CSV/TSV, and one dot-separated `.moves` file with a single puzzle context.
- Derived client fields: replay/reached proof hash, solution length, UUIDv7, UTC timestamp, semantic idempotency, canonical JSON and deterministic gzip.
- Receipt manifest contains only endpoint origin, input/part hashes, sizes/states, submission/idempotency/status/GitHub URLs and terminal safe states. Submitted envelopes are not persisted there.
- Same-origin status enforcement; after expected D1 cleanup a 404 is verified using the immutable expected GitHub raw URL.

## Verification

- Python CLI/plugin unit suite: 17/17 passed.
- Existing Worker schema/config suite: 72/72 passed.
- Existing Worker/Queue/GitHub suite: 138/138 passed.
- TypeScript `tsc --noEmit`: passed.
- Plugin manifest validator: passed.
- Skill validator: passed.
- Documented `init` and three template preflights: passed; v1 CSV and full JSON each produced one part, v2 full JSON produced one part.
- Secret scan found no credential literal; the only security terms are rejection logic and explanatory documentation. Test-only localhost/evil URLs remain limited to tests and the hidden environment-gated endpoint override.
- `git diff --check`: passed.

## Live staging result

No live receipt was created in this run. Direct Windows TLS to `workers.dev` failed before HTTP (`HTTP_TRANSPORT`); the configured local HTTP proxy accepted CONNECT but its TLS handshake also failed. A private CPU Kaggle smoke was prepared but the execution environment rejected that third-party code/artifact upload, so it was not pushed and no workaround was attempted. This is an external verification gap, not a claimed live success. The same endpoint and both schema routes remain covered by the 210 passing Worker tests and prior checked-in live v1/v2 staging evidence.

## Artifacts

- Plugin: `plugins/cayleypy-results-publisher/`
- Russian guide: `plugins/cayleypy-results-publisher/references/HUMAN_GUIDE_RU.md`
- Agent protocol: `plugins/cayleypy-results-publisher/references/AGENT_PROTOCOL.md`
- CLI: `plugins/cayleypy-results-publisher/scripts/cayleypy_submit.py`
- Templates: `plugins/cayleypy-results-publisher/templates/`
# Final Kaggle and GitHub verification

- Private Kaggle kernel: trydotatwo/cayleypy-results-publisher-private-smoke, version 6, status COMPLETE.
- Kaggle v1 submission: accepted=1, published=1, submission 019fc87f-8be8-7f40-91c0-fc7cd69a4a2a.
- Native v2 submission: accepted=1, published=1, submission 019fc880-0080-7787-bfb7-98fcad07ccce.
- GitHub ingest/staging contained the v1 result at results/v1/toy-cayley/cube_3-3-3/1/2026-08-03/019fc87f-8be8-7f40-91c0-fc7cd69a4a2a.json (blob 0fb3fe196d168612a3814902229668523ccd400c).
- GitHub ingest/staging contained the v2 result at data/v2/slurm/toy-cayley/cube_3-3-3/2026-08-03/019fc880-0080-7787-bfb7-98fcad07ccce.json (blob 15b2464f7377676fbdee3fcbbfee7fcbca36e3d2).
- Earlier diagnostic versions exposed and fixed two client defects: Cloudflare rejected Python's default urllib signature (1010), and native-v2 run_id did not match provenance.run_id. The client now sends a stable public User-Agent, handles and retries transport timeouts safely, and reports sanitized HTTP status without secrets.
- Kaggle output and receipts are retained locally under test_results/cayleypy_publisher_kaggle_smoke/output_v6.
