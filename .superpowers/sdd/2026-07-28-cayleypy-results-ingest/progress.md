# SDD ledger — plan: docs/superpowers/plans/2026-07-28-cayleypy-results-ingest.md
Baseline: fa31690; 23 tests passed.
Task 1: fix round 1/5 (formats/raw-byte/maxBytes addressed; runtime gate still open; commits b6c522a..90ace02)
Task 1: fix round 2/5 (published exact pin selected; lock/runtime still open; commits 90ace02..e1d8ff3)
Task 1: fix round 3/5 (compatible pin, lockfile, 11/11 tests and typecheck addressed; commits e1d8ff3..eb568a0)
Task 1: complete (commits fa31690..eb568a0, review clean; private Kaggle v3 payload byte-matched)

Task 2: fix round 1/4 (durable receipts; commit e020a67)
Task 2: fix round 2/4 (concurrency hardening; commit d747dd4)
Task 2: fix round 3/4 (stale recovery; commit dde1cc9)
Task 2: fix round 4/4 (mode/migration review and lossless pause; commits 9a966d5..d50d7bb)
Task 2: complete (commits eb568a0..d50d7bb, review clean; private Kaggle v16 payload byte-matched)

Task 3: fix round 1/3 (behavior-first Worker RED; private Kaggle v17)
Task 3: fix round 2/3 (Ajv Worker-pool module-resolution diagnosis; private Kaggle v18..v19)
Task 3: fix round 3/3 (exact deep-entry optimization and corrected test harness; private Kaggle v20)
Task 3: implementation complete (this standalone commit; private Kaggle v21 payload byte-matched; parent review pending)
