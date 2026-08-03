# Megaminx native autotune resume hardening ? 2026-08-03

## Live evidence

- A100x8 job `33357` found an exact-scale bootstrap boundary near `770,874,022` but selected finalists with larger shard reserve that OOMed.
- Resumed job `33360` correctly reused the checkpoint and failed closed with `complete=0`; solve did not start.

## Corrections

- Validate every maximum-beam probe on calibration puzzles 900, 950, and 1000.
- Remove all-OOM candidate groups from successive-halving.
- Finalize a winner only after all final repetitions pass.
- If the fast winner fails final validation, validate and select the exact-scale bootstrap seed.
- Scope measured final evidence by profile power, exact beam, and config id.
- Record `maximum_stable_beam` and reject a requested beam above it even when it maps to the same rounded power.

## Verification

```text
python -m pytest tests/portable -q
210 passed, 3 skipped in 5.11s
```
