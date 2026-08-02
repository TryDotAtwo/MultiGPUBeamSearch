# Hardware-specific depth-8 autotuner v2

Date: 2026-08-02

## Requirements

- Re-tune every runtime control for each exact GPU/hardware tuple.
- Score a fully filled frontier at depth 8; early depths are unscored.
- Tune final materialization chunk only in a deliberately small range.
- Find fast configurations with meaningful memory reserve.

## Implementation

- Added opt-in native `autotune_depth_done` records without enabling general debug instrumentation.
- Benchmark workflow accepts only completed depth 8 with global frontier at least the requested beam.
- Probe score uses depth-8 critical-path time and global frontier throughput; setup and depths 1-7 are excluded.
- Hardware identity remains keyed by family, SM, VRAM, world size, driver, solver, model, and release.
- A 25-point deterministic mixed design jointly covers b_micro, concurrency, ring slots, shards/capacity, Stream 4 controls, and final chunk values 32768/65536/98304/131072.
- Beam anchors run maximum-first.
- All three successive-halving rounds now rank survivors.
- Final ordering chooses minimum peak VRAM among configurations within 3% of fastest median.
- Hard VRAM limit changed to 85%, preserving at least 15% reserve.

## Verification

- Full portable suite: 196 passed, 3 skipped before documentation-only edits.
- Focused depth-8 workflow/probe tests validate full and unfilled frontier behavior.
- Git diff check passes.

Native CUDA compilation and six-architecture archive gates are delegated to the release workflow after push.