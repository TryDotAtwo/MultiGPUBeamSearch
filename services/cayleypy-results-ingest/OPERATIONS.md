# CayleyPy results ingest operations

## Public boundary

POST /v1/results is intentionally anonymous so public Kaggle notebooks can publish without a secret. Submitted author fields are **claimed**, not identity-verified; dashboards and public indexes must preserve that label.

Submission status URLs are bearer-like: do not publish, commit, or include them in logs. The Worker applies a D1-backed per-IP GET limit and returns only { "error": "rate_limited" } on exhaustion.

## Dead-letter replay

A dead letter is never automatically replayed. Raw R2 input is retained and is the recovery source.

The operator-only replayDeadLetters(env, options) primitive has this contract:

1. Default call is dry-run: it selects the oldest bounded page (default 25, maximum 100) and mutates nothing.
2. Apply requires the explicit apply: true flag.
3. Every selected row is checked with RAW_RESULTS.head(raw_r2_key) before mutation. Missing raw input is skipped and remains dead_letter.
4. Successful apply performs only dead_letter -> retryable with safe_error=operator_replay_pending. It does not enqueue directly.
5. The normal scheduled recovery later requeues eligible retryable rows. This prevents an automatic DLQ loop.

Before apply, snapshot the selected D1 rows and R2 keys; alert on skipped raw objects or any remaining dead letters. Do not use a public HTTP endpoint for replay.