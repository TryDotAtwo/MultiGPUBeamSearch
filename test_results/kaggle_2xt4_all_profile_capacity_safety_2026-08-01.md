# Kaggle 2xT4 all-profile capacity safety audit — 2026-08-01

## Failure reproduced

Kaggle version 4 on puzzle 999, output1 p18, failed at depth 10 with `code=3002`: one physical shard had `existing=147301`, `available=49307`, and received `raw_count=49401`, leaving 94 candidates without capacity while its A/B sibling was processing.

## Fix

For every derived MLP and piece-transformer profile, require:

`SHARD_CAPACITY_CANDIDATES >= align(STREAM3_BATCH_CANDIDATES + STREAM4_BATCH_CANDIDATES + STREAM4_TRIGGER_CANDIDATES, 1024)`

The three terms cover one complete incoming Stream3 receive, one resident clean Stream4 batch, and dirty candidates immediately below the Stream4 trigger. The sum is deliberately conservative by at least one candidate.

The failed p18 tuple changes from 196608 to 393216 candidates per physical shard. `GLOBAL_SPILL_CAPACITY=0` remains unchanged for A/B mode.

## Coverage

- MLP registry: output1 p16..p25 and output_move_count p16..p25.
- Piece-transformer registry: output_move_count p16..p26.
- Total audited anchors: 31.
- Arbitrary beams inherit the same derivation after nearest-profile selection.

## Local verification

- Baseline: `13 passed`.
- Red regression: p18 and registry audit both failed against the old derivation.
- Focused post-fix profile suite: `15 passed`.
- Full repository Python suite: `306 passed`.`n- All checked-in Kaggle notebooks: JSON parse and Python-cell AST parse passed.`n- `git diff --check`: passed.

## Remaining hardware gate

The largest additional resident CandidateMeta allocation among current MLP anchors is about 460 MiB per rank for output1 p25. Static correctness is verified locally; real T4 allocation and end-to-end completion still require a new private Kaggle 2xT4 run before calling the adjusted profiles hardware-validated.