# Kaggle 2xT4 capacity-safe rollout to five notebooks — 2026-08-01

## Scope

Regenerated and checked:

1. Main universal checkpoint notebook.
2. 444 piece-Transformer example.
3. Megaminx output-24 example.
4. IHES output-1 example.
5. Professor Tetraminx output-1 example.

All five pin solver commit `6d4471c4ab03c528fd7ce1e15c0cc9db11774833`, which contains the profile-wide A/B capacity bound.

## Profile capacity audit

All 31 Kaggle 2xT4 anchors are covered: MLP output1 p16..p25, MLP output_move_count p16..p25, and piece Transformer p16..p26.

For every selected profile and arbitrary beam using that profile:

`SHARD_CAPACITY_CANDIDATES >= align(STREAM3_BATCH_CANDIDATES + STREAM4_BATCH_CANDIDATES + STREAM4_TRIGGER_CANDIDATES, 1024)`

The failed output1 p18 case derives 393,216 candidates instead of 196,608. Runtime throughput knobs (B_MICRO, concurrency, ring slots, shard count, Stream4 batch/trigger/sort slots, final materialization chunk) are unchanged.

## Resource and speed assessment

The change only enlarges resident candidate capacity. It does not add kernels, synchronization, transfers, or candidate work. The largest MLP allocation increase is approximately 460 MiB per rank at output1 p25; lower anchors add 24–230 MiB per rank. Transformer p26 retains its previously measured runtime tuple and the 88,064 final materialization chunk.

Static capacity correctness does not replace a real T4 acceptance run. Until the new commit is pushed and the five Kaggle versions run, the updated memory layout is locally verified but not newly hardware-accepted.

## Verification

- Focused builder/profile suite: 31 passed.
- Five checked-in notebooks: JSON parse, Python AST parse, and exact solver-pin count passed.
- Full Python suite: `311 passed`.
- Deterministic regeneration: all five generated notebook bytes match checked-in artifacts.
- `git diff --check`: passed.