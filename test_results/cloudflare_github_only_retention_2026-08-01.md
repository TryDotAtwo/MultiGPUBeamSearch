# Cloudflare transient-storage cleanup verification — 2026-08-01

## Contract

- GitHub is the only long-term results store.
- R2 raw payload, D1 submission row, and Durable Object pending key remain while GitHub publication can still fail.
- After a confirmed GitHub commit, cleanup order is R2 -> D1 -> Durable Object.
- A failed GitHub request keeps the validated D1 row, R2 payload, pending key, and a future alarm for retry.
- A partial cleanup is retryable: a staged D1 row is recognized and cleaned without a second GitHub commit.

## TDD evidence

The two success tests first failed because the existing implementation retained D1 and R2 after staging. After implementation:

- schema/config tests: 56 passed;
- Worker/Miniflare tests: 122 passed;
- targeted GitHub writer tests: 12 passed;
- TypeScript typecheck: passed.

No CUDA/C++ or beam-search architecture was changed.
## Historical backlog

A post-deploy audit found 489 legacy staged rows. Added a normal-mode scheduled cleanup page of 100 rows/minute. Full verification now passes 56 schema/config and 123 Worker tests plus typecheck.

Legacy validated rows are re-enqueued to GitHubWriter in bounded pages of 100/minute; they remain in R2/D1 until GitHub preflight or commit succeeds. Final suite: 56 schema/config + 124 Worker tests and typecheck.

## Free-plan publication budget

Live recovery exposed a Cloudflare Free boundary: a 100-record writer flush performs more than 50 external GitHub subrequests and therefore retries the whole batch forever. The publication batch is now capped at 40 records, leaving room for token, ref, tree, commit, and ref-update calls while the scheduled recovery may still enqueue 100 ids per minute.

TDD evidence:

- RED: 41 validated records produced 41 content preflights instead of the allowed 40;
- GREEN: focused suite passed 56 schema/config + 13 writer tests;
- full suite passed 56 schema/config + 125 Worker tests and TypeScript typecheck.
