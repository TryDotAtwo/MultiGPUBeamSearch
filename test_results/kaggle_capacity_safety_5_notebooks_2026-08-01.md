# Kaggle 2xT4 capacity-safety verification (2026-08-01)

## Code gates

- Branch: `codex/profile-capacity-safety`
- Notebook solver pin: `039a4fb9c699bda644efbb0f8ee7a5b51efbb5d2`
- Full Python suite: `312 passed in 35.35s`
- Result-envelope suite: `54 passed in 30.18s`
- Notebook builder tests: `14 passed in 1.12s`
- Generated notebook gate: all five files passed JSON parsing, combined Python AST parsing, and exactly-one-pin validation.
- Capacity audit: all 31 MLP and piece-transformer anchors satisfy `align(stream3_batch + stream4_batch + stream4_trigger, 1024)`.

## Live Kaggle runs

| Notebook | Version | Kaggle status | Solve | Capacity | Publish |
|---|---:|---|---:|---:|---|
| Universal checkpoint | 7 | COMPLETE | setup landing | n/a | n/a |
| Cube 4x4 | 44 | COMPLETE / success | 9.228 s | 51,200 | HTTP 400 (stale staging schema) |
| Megaminx | 9 | COMPLETE / success | 6.747 s | 393,216 | HTTP 202 |
| IHES | 7 | COMPLETE / success | 9.591 s | 393,216 | HTTP 202 |
| Professor Tetraminx | 6 | COMPLETE / success | 9.792 s | 393,216 | HTTP 202 |

All four configured examples solved one puzzle on exactly two Tesla T4 GPUs. Focused scans of rank/combined logs found no code 3002, overflow, OOM, traceback, or fatal error.

Cube 4x4 used the measured piece-transformer p16 profile. Its derived bound was exactly `18,432 + 16,384 + 16,384 = 51,200` candidates. The solve and local strict result-envelope validation succeeded; only the external staging Worker rejected the new transformer manifest with HTTP 400.

## Ingest staging diagnosis

The isolated ingest worktree at commit `7b0cab2` already contains transformer-manifest support. Its current gate passed:

- schema tests: 56 passed;
- Worker tests: 122 passed;
- TypeScript typecheck: passed;
- Wrangler 4.115.0 staging dry-run: passed.

The live deploy was not performed because this non-interactive session has no `CLOUDFLARE_API_TOKEN`. Wrangler stopped before mutation. A staging deploy of the already-tested ingest revision is required before repeating Cube 4x4 publication and expecting HTTP 202.
