# Puzzle 0 Full Path Score-Mapping Trace

## Context
- User risk hypothesis: Stream1 output indices `[0..23]` may not match `p900.json` move indices `[0..23]`.
- Required diagnostic: compare CUDA Stream1 scores and host reference-forward scores for the tracked parent, all 24 moves, with `q`, rank, and `move_name`.
- Target run: puzzle `0`, tracked path length `55`, beam `2**24`.

## Code Changes
- `GeneratedTrackResult` now carries the copied parent `State128` and all 24 CUDA score keys for the tracked parent row.
- `production_runner` now computes a host reference-forward pass from exported fp16 Stream1 weights for the copied parent state.
- `production_runner` logs:
  - `track_solution_move_score_summary`
  - `track_solution_move_score` for all 24 moves.
- `DepthDispatchState` now stores all tracked Stream4 events for the tracked shard:
  - `after_stream3`
  - every `stream4_input`
  - every `stream4_output`
- `production_runner` logs every event as `track_solution_stream4_event`.
- Added `BEAM_STOP_AFTER_TRACKED_MISSING_EXTRA_DEPTHS`; if a tracked prefix first disappears, the run continues the requested number of extra depths and then stops.

## Kaggle Diagnostic Config
- `START_PUZZLE_ID = 0`
- `PUZZLE_COUNT = 1`
- `DEPTH_LIMIT = 80`
- `BEAM_WIDTH = 2**24`
- `STOP_AFTER_TRACKED_MISSING_EXTRA_DEPTHS = 2`
- `KNOWN_SOLUTION_PATHS[0]` set to the user-supplied path.

## Verification
- `git diff --check` passed.
- Local Windows CUDA build remains unavailable because the local CMake generator reports `No CUDA toolset found`.
