# Architecture

## Scope of the current implementation

Current target: correct CUDA/C++ multi-GPU beam-search depth loop on Kaggle 2×T4 and local CUDA machines.
Transformer Engine backend is intentionally not part of the Kaggle/T4 correctness pass. TorchScript neural scorer ensemble is part of the correctness pass.

## Static buffers

Python allocates all GPU buffers through `torch` and passes pointers to C++:

- `beam_current [N_LOCAL, 120] uint8`: current frontier states.
- `current_active_flags [N_LOCAL] uint8`: valid entries in current frontier.
- `next_state_pool [K_WORK, 120] uint8`: next frontier candidate states.
- `next_meta [K_WORK * sizeof(BeamMeta)] uint8`: parent/action/score metadata.
- `hash_table [HASH_CAPACITY * sizeof(HashSlot)] uint8`: owner-local dedup table.
- `active_flags [K_WORK] uint8`: valid entries after insertion/prune.
- `score_ring int16`: action-major score ring; C++ treats memory as `uint16_t`. Neural scorers write `[B, 24]` into this ring through a layout-conversion kernel.
- `send_buckets/recv_buckets`: fixed-size NCCL candidate buckets.
- `local_hist/global_hist`: score histograms.
- `threshold_cell`: `valid, threshold_q`.
- `counters`: debug counters.
- `beam_status`: current size, compacted size, found flag, graph flag.

## Per-depth pipeline

One depth step is:

```text
clear hash/table/hist/next flags
check current frontier for central-state
for each micro-batch:
  stream1: one of N inference lanes scores actions for active current states
  stream2: waits score_ready[slot], applies real puzzle move, hashes, owner-routes, local insert/update, bucket pack
  stream3: NCCL fixed all-to-all for remote buckets
  stream2: ingest received buckets into owner-local hash table
  stream3: periodic global histogram all-reduce and threshold computation
final global threshold update
stream2: threshold prune
stream2: compact active next_state_pool -> beam_current
stream3: all-reduce found flag
```

The CUDA Graph captures this complete depth step. Graph capture starts on the root stream; inference lanes, `stream_ingest`, and `stream_net` join capture through CUDA events. Ring reuse is protected by `score_consumed[slot]` and `net_consumed[slot]`, so graph capture does not serialize the three-stream pipeline.

## Stop condition

`BeamEngine.search(max_depth)` performs host-level per-depth control:

```text
if found before expansion: return found at current depth
if depth == max_depth: return not found
launch one captured depth step
repeat
```

The `found` flag is set on GPU when an inserted candidate equals uploaded `central_state`. In multi-GPU mode `found` is reduced through NCCL with `ncclMax`.

## Puzzle data integration

`data_loader.py` loads:

- `data/puzzle_info.json`;
- `data/test.csv`;
- `data/sample_submission.csv`.

Action order is fixed:

```text
-B, -BL, -BR, -D, -DL, -DR, -F, -FL, -FR, -L, -R, -U,
B, BL, BR, D, DL, DR, F, FL, FR, L, R, U
```

CUDA move semantics:

```cpp
dst[i] = src[action_permutation[action][i]]
```

CPU reference uses the same rule:

```python
next_state = state[perm]
```

## Known non-goals for this pass

- Transformer Engine inference.
- Full path reconstruction across depths/ranks.
- Exact stable top-K tie handling with global ordering.
- 100×H100 Slurm performance tuning.

## Neural scorer

See `docs/NEURAL_SCORER.md`. Current implemented neural path is `torchscript_ensemble`: arbitrary PyTorch model must be exported to TorchScript and must return `[B, 24]` scores. Default provided scorer is `120 -> 1024 -> 256 -> 24`.
