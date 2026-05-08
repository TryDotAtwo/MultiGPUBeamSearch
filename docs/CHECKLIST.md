# Correctness checklist

## Required before performance work

- [x] `puzzle_info.json` loader validates central state and all 24 permutations.
- [x] `test.csv` loader validates state shape and dtype bounds.
- [x] CPU action reference uses the same `dst[i] = src[perm[i]]` rule as CUDA.
- [x] Real action table is uploaded to CUDA constant memory.
- [x] Central state is uploaded to CUDA constant memory.
- [x] `beam_current` is initialized from a real puzzle state, not zero-filled dummy data.
- [x] Current frontier has `current_active_flags`.
- [x] Per-depth step clears next-step hash/table/hist/flags.
- [x] Hash insert publishes metadata before hash publication through BUSY → committed hash.
- [x] `HashSlot` uses `uint32_t best_key`; no `atomicMax` over a `uint16_t` field.
- [x] Histogram increments only for inserted/updated unique states.
- [x] Stream2/Stream3 threshold update is event-synchronized.
- [x] Next frontier is compacted back into current frontier.
- [x] Search stops on central-state or `max_depth`.
- [x] CUDA Graph capture is enabled by default for correctness tests.
- [x] Kaggle notebook avoids source monkey-patches.
- [x] 2-GPU test requires `remote_packed > 0`.

## Still open

- [ ] Production neural/TE scoring backend.
- [ ] Full path reconstruction.
- [ ] Exact global top-K with deterministic tie ordering.
- [ ] Large-width memory/throughput tuning after correctness is stable.
