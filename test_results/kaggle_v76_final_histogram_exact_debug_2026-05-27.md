# Kaggle v76 Final Histogram Exact Debug

## Run
- Date: 2026-05-27
- Kernel version: 76
- Commit: `01d4867`
- GPU: T4 x2
- Beam width: `67108864`
- Depth limit: `16`
- Result: `return_code=-6`
- Seconds: `40.02146702400023`
- Logs: `test_results/kaggle_v76_output/`

## Failure
- Message: `final selected count does not match score phase counts`
- Final threshold: `27607`
- Global keep: `67108864`
- Final exchange exact counts: `global_less=67255361`, `global_equal=14883`
- Histogram counts: `hist_less=67099219`, `hist_equal=17064`

## New Diagnostic Signal
- Rank 0 summary:
  - `local_hist_total=35425566`
  - `exact_host_total=35425566`
  - `local_hist_less=33557119`
  - `exact_host_less=33760737`
  - `exact_local_less=33701675`
  - `diff_host_less=+203618`
  - top local bin diff: `score=0 local_hist=0 exact_host=4886157`
- Rank 1 summary:
  - `local_hist_total=35849088`
  - `exact_host_total=35849088`
  - `local_hist_less=33542100`
  - `exact_host_less=34043045`
  - `exact_local_less=33553686`
  - `diff_host_less=+500945`
  - top local bin diff: `score=0 local_hist=0 exact_host=4323292`

## Shard Localization
- Rank 0 mismatching shards:
  - shard 3: `active_less=4222232`, `exact_less=4366453`, `diff_less=+144221`, `min_score=0`
  - shard 4: `active_less=4185103`, `exact_less=4244500`, `diff_less=+59397`, `min_score=0`
- Rank 1 mismatching shards:
  - shard 3: `active_less=4216571`, `exact_less=4225220`, `diff_less=+8649`, `min_score=0`
  - shard 4: `active_less=4144148`, `exact_less=4636444`, `diff_less=+492296`, `min_score=0`

## Interpretation
- Stream4 active histogram totals still match `clean_count`; `total_mismatch_shards=0`.
- Host exact scan after final score-count kernels sees millions of `score_key=0` entries that Stream4 histogram does not contain.
- GPU final less count is also different from host exact count, meaning candidate memory changes during or after the GPU score-count phase.
- The leading root cause is final phase scratch overlay: `final_keep_flags`/phase-2 final buffers likely overlap `survivor_shard`, so final score-count kernels write flags into live `CandidateMeta` storage while still reading `survivor_shard`.
- This explains the synthetic `score_key=0` spike and the mismatch between histogram-derived threshold and exact final selection.

## Next Fix Candidate
- Preserve `survivor_shard` as live input during final phase 2.
- Move final keep/count/offset scratch so phase-2 selection buffers do not overlap `LayoutStreamsView::survivor_shard` until after exact selection has materialized selected candidates.
- No code fix was applied in this report; the run was diagnostic-only as requested.
