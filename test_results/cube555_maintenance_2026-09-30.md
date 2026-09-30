# Cube555 downstream maintenance gate — 2026-09-30

Confirmed before patch: stateless scheduled events repeatedly exceeded CPU at10ms; publication DO alarms failed with safe github_writer_retryable. All2000 original semantic identities accepted in D1; latest verified original blobs330main/610staging, not complete publication.

Cron now delegates only scheduledTime to the existing publication DO internal maintain RPC. Same normal-mode guard,100staged cleanup/100validated enqueue/stale recovery limits and transitions preserved. No heavy edge fallback. Actual DO RPC, state preservation, malformed timestamp, paused edge/no-binding behavior tested. GitHub failure diagnostics output only fixed event/operation/method/status and bounded numeric retry/rate headers; error bodies/tokens/paths not logged.

Local gate:72schema +156runtime testsPASS; tscPASS; Wrangler dry-runPASS, all bindings unchanged. Expected aborted-pipe messages in existing canceled-ingress tests do not indicate gate failure. Live Cron/GitHub exact downstream verification remains pending at this commit.

Long acceptance official CLI RUNNING; one own slot. Current publicv13 runtime pin10824003 remains unchanged (server maintenance independent of notebook/native runtime).

Live follow-up: deployed sourcec37ce051 Cloudflareversionc2fd6e8d-fb23-46e5-a2ab-c53739d0a587. Two Cron events outcomeok CPU0ms/wall~23.6s; maintain RPC outcomesok CPU106/98/10ms; writer alarm outcomeok CPU176ms/wall131904ms. Onealarm canceled observed, no cause inferred. Exact immutableGit tree comparison now650/2000main,770/2000staging,0mismatch. Full drain/promotion stillpending; no all-ready claim.
