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