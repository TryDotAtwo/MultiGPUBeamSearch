# Kaggle v75 T4x2 2^26 Failure

## Run
- Date: 2026-05-27
- Kernel: `trydotatwo/cayley-beam-gpu-runner`
- Kaggle version: 75
- GPU: T4 x2
- Beam width: `67108864`
- Depth limit: `16`
- Result CSV: `test_results/kaggle_v75_output/beam_run_results.csv`
- Rank logs: `test_results/kaggle_v75_output/run_logs/`

## Result
- `return_code=-6`
- `seconds=40.029106731999946`
- Failure phase: multi-rank final selection
- Failure message: `final selected count does not match score phase counts`
- Memory-budget failure: not reproduced
- Stream3 double-buffer overflow `code=3002`: not reproduced in captured tail
- Global spill: `0` in captured final-drain tail

## Final Selection Evidence
- `final_threshold=27609`
- `global_keep_count=67108864`
- Histogram summary: `hist_total=71868331`, `hist_less=67087422`, `hist_equal=36345`
- Exact survivor scan: `exact_global_less=67216299`, `exact_global_equal=31755`
- Mismatch: `diff_global_less=+128877`, `diff_global_equal=-4590`
- Histogram total invariant: `hist_invariant_ok=1`
- Exact threshold invariant: `exact_invariant_ok=0`
- Rank 0 local mismatch: `diff_local_less=+14324`, `diff_local_equal=-2167`
- Rank 1 local mismatch: `diff_local_less=+114553`, `diff_local_equal=-2423`
- Final exchange exact counts: `less_counts=[33564055,33652244]`, `equal_counts=[16078,15677]`

## Interpretation
- The forced A/B config and phase-layout memory repack were sufficient to pass runtime allocation and reach final selection with `BEAM_WIDTH=2^26`.
- The current blocker is not GPU memory capacity.
- The current blocker is a mismatch between histogram-derived threshold counts and exact selected-candidate score counts.
- `hist_invariant_ok=1` and `total_mismatch_shards=0` show active histogram totals equal clean shard totals.
- `exact_global_less > global_keep_count` shows final exact score comparison finds too many candidates strictly better than `final_threshold`.
- The likely fault class is score-key inconsistency between Stream4 histogram population and final exact survivor scan, or a threshold polarity/scale mismatch in one of those paths.

## Next Debug Target
- Add or inspect per-score-bin diagnostics near `final_threshold`.
- Compare Stream4 histogram bin counts for `[final_threshold-2, final_threshold+2]` against exact candidate score bins from `survivor_shard + clean_count`.
- Keep architecture unchanged until the mismatch source is localized.
