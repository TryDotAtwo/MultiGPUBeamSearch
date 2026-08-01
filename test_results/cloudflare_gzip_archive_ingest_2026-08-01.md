# Cloudflare gzip archive ingest gate - 2026-08-01

Contract:

- `Content-Type: application/gzip`;
- compressed request maximum: 32 MiB;
- decompressed JSON maximum: 64 MiB;
- maximum results per archive: 2,000;
- legacy `application/json` remains accepted with the existing 4 MiB bound.

Evidence:

- RED: the Worker returned HTTP 415 for one gzip archive containing 101 valid results.
- GREEN: the same real Worker request returned HTTP 202 and 101 receipts.
- Full schema/config suite: 57 passed.
