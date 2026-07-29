# CayleyPy results-ingest canonical contract port (2026-07-29)

- Source contract: public commit `458d0fa`; schema SHA-256 `bcb28d05454a8f8098c3312d53c26c2a16d4075c5c292290468cb391cd1ff662`, golden SHA-256 `6c5d241574c0ca0f859791d39ef625394f4e64d7d4e7986a04ce968f3f2ad172`.
- Port retains the existing pinned Cloudflare package/lock stack unchanged and does not alter Task 2/3 persistence, migrations, Queue behavior, CUDA, or beam code.
- Local npm evidence is blocked by npm CLI `Exit handler never called` (isolated cache log under `D:\100XH100\test_results\cayleypy_results_ingest_npm_cache`). The private Kaggle exact-stack gate is regenerated for the canonical contract instead of weakening evidence.
- Pending gate must demonstrate schema, Worker, and typecheck against the payload checksum before this report is marked green.
## Green gate

Private Kaggle exact-stack gate v34 passed all 14 commands: schema 6/6, receipt plus Worker 70/70 twice, and typecheck. Payload and post-install hashes matched; compatibility warning scan was empty. The canonical schema and golden SHA-256 remained cb28d05454a8f8098c3312d53c26c2a16d4075c5c292290468cb391cd1ff662 and 6c5d241574c0ca0f859791d39ef625394f4e64d7d4e7986a04ce968f3f2ad172.
