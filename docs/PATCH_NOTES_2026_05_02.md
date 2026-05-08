# Patch notes 2026-05-02

## Changed files

- `beam_engine_common.hpp`
- `beam_kernels.cu`
- `beam_engine.cpp`
- `beam_engine.py`
- `data_loader.py`
- `scripts/kaggle_correctness_check.py`
- `scripts/t4_sizing.py`
- `notebooks/kaggle_2xt4_debug.ipynb`
- `notebooks/notebookaafc902d8e (4).ipynb`
- `docs/ARCHITECTURE.md`
- `docs/CHECKLIST.md`
- `docs/KAGGLE_T4_DEBUG.md`
- `docs/INDEX.md`

## Main fixes

1. Removed zero-filled fake frontier as the default test path.
2. Added real `puzzle_info.json` action table upload to CUDA constant memory.
3. Added real `central_state` upload to CUDA constant memory.
4. Added `current_active_flags` for valid current-frontier entries.
5. Added `beam_status` for current size, compacted size, found flag and CUDA Graph captured flag.
6. Added publication-safe hash insertion using `HASH_BUSY` and committed slot flags.
7. Replaced invalid `atomicMax` on a `uint16_t` field with `HashSlot.best_key uint32_t`.
8. Histogram now updates only on unique insert or real score update.
9. Added event synchronization between stream2 histogram writes and stream3 threshold update.
10. Added compaction from `next_state_pool` back to `beam_current`.
11. Added `BeamEngine.search(max_depth)` with stop-on-central-state/max-depth semantics.
12. Made CUDA Graph the default correctness path.
13. Replaced notebook monkey-patches with a real correctness runner.
14. Added 1-GPU and 2-GPU correctness tests for shallow generated cases and `test.csv` one-step expansion.

## Explicitly not solved in this patch

- Full path reconstruction across depths/ranks.
- Transformer Engine backend.
- Exact deterministic global top-K extraction.
- Large-scale performance tuning for 100×H100.
