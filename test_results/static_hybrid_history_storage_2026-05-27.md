# Static Hybrid History Storage 2026-05-27

- Date: 2026-05-27
- Scope: CPU candidate history storage only.
- Code path: `tools/production_runner.cu::CpuCandidateHistory`
- Kaggle config files: `kaggle/cayley-beam-gpu-runner.ipynb`, `kaggle/beam_kernel.ipynb`

## Implemented

- Added `BEAM_HISTORY_MODE=static_hybrid`.
- Kept `HISTORY_SLOT_COUNT` as pinned `CandidateMeta[32B]` GPU-to-CPU copy slots.
- Added startup budget accounting for pinned slots, per-slot history staging, RAM `HistoryEntry[16B]` arena, and disk `HistoryEntry[16B]` arena.
- Added `BEAM_HISTORY_RAM_BYTES`, `BEAM_HISTORY_DISK_BYTES`, and `BEAM_HISTORY_DISK_PATH`.
- Added disk-first dense `HistoryEntry` storage in one per-rank static disk arena file.
- Added RAM fallback when configured disk arena is exhausted or a disk write fails.
- Kept no-prune semantics: `parent_idx` is not remapped.
- Kept distributed reconstruction semantics: `route_packed.source_rank` selects rank, `parent_idx` selects local depth entry.

## Kaggle Defaults

- `HISTORY_MODE=static_hybrid`
- `HISTORY_SLOT_COUNT=2`
- `HISTORY_RAM_BYTES=28 * 1000**3`
- `HISTORY_DISK_BYTES=49 * 1000**3`
- `HISTORY_DISK_PATH=/kaggle/working/history_arena`

## Local Verification

- `git diff --check`: passed.
- Docker build: `production_runner history_tests`: passed.
- Docker ctest: `history_tests`: passed.
- Local production smoke with `static_hybrid` did not reach history initialization because the current local Docker GPU budget rejected all runtime configs before history setup: `no runtime config fits GPU/final-layout budget`.

## Pending

- Push to GitHub `main`.
- Launch Kaggle T4x2 validation.
- Inspect logs for `candidate_history_*` budget lines and `history_bytes_stored_ram/history_bytes_stored_disk`.
