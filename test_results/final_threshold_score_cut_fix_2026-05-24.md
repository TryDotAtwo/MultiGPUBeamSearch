# Final Threshold Score Cut Fix 2026-05-24

- issue: Kaggle T4 diagnostics showed `final_threshold=4294967295` while `final_candidate_count=16820224`.
- root_cause_1: `SCORE_SCALE=1024` raised `SCORE_MAX_KEY` to `307200`, but Stream4 score histogram radix sort still used `end_bit=16`.
- root_cause_1_effect: score keys above `65535` were sorted only by low 16 bits; `ReduceByKey` saw split groups; histogram bins were overwritten with partial counts; final threshold selection undercounted survivors.
- root_cause_2: final exact cap used layout traversal after `score_key <= final_threshold`; when threshold was `UINT32_MAX`, all candidates passed and the cap preserved first-by-layout candidates.
- fix_1: `cuda/stream4.cu` and CUB temp sizing in `cuda/static_memory.cu` now use 32-bit radix sorting for score histograms.
- fix_2: `cuda/threshold.cu` final selection now emits `score_key < final_threshold` candidates before the threshold-score tail, then caps the threshold tail to the remaining aligned beam slots.
- test_update: `tests/threshold_cuda_tests.cu` now includes a regression where two threshold-score candidates appear before a lower-score candidate; expected output keeps the lower-score candidate.
- local_verification: `git diff --check` passed.
- local_blocker: host `cmake --build build-docker` failed because `CMakeCache.txt` was created inside `/work/build-docker`; Docker daemon was unavailable on `npipe:////./pipe/dockerDesktopLinuxEngine`, so CUDA tests were not executed locally in this turn.
