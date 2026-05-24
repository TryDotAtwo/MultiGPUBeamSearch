# Tracked Solution Generated-Candidate Logging

## Context
- Target failure: puzzle 9 tracked path survives final filtering at depth 5 with `prefix_len=6`, but depth 6 `prefix_len=7` is absent before final filtering.
- Known implication: final filter is not the first loss point for `prefix_len=7`.

## Change
- Added `GeneratedTrackRequest` and `GeneratedTrackResult` to dispatcher API.
- Added host-side diagnostic scan of score/hash rings after Stream1/2 ring completion and before Stream3 graph launch.
- Added `track_solution_generated` log line in `production_runner`.

## Logged Fields
- `expected_parent_idx`: previous depth final frontier index used as current depth parent.
- `expected_move`: move expected for the tracked path at current depth.
- `ring`, `ring_slot`, `job`, `parent_base`, `parent_local`, `count`.
- `payload_id`, `score_ring_offset`.
- `score_key`, `current_threshold`, `threshold_pass`, `threshold_margin`.
- `hash_lo`, `hash_hi`, `owner`, `shard`.

## Interpretation
- `track_solution_generated found=0`: tracked parent was not present in current frontier or scheduler mapping.
- `track_solution_generated found=1 threshold_pass=0`: Stream1 scored the tracked child above current threshold, so Stream3 threshold should prune the path.
- `track_solution_generated found=1 threshold_pass=1` plus missing `track_solution_prefinal`: loss is after Stream3 threshold, in Stream3 dedup/collector, spill handling, or Stream4 threshold/dedup.
- `track_solution_prefinal found=1` plus missing `track_solution_prefix`: loss is final filter/load-balance.

## Verification
- `git diff --check` passed.
- Local Windows CMake build is blocked by missing CUDA toolset: `No CUDA toolset found`.
- Docker verification is currently blocked by unavailable Docker Desktop in this session.
