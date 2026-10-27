# Cube555 2.5M beam: calibration failure and repair

- Long Kaggle kernel: `trydotatwo/cube555-2-5m-beam-long-solution-2xt4`, version 1, status `ERROR`.
- Failure reported from the Kaggle traceback: `calibration_stats.select()` -> `estimate(baseline, samples)` -> `ValueError: unstable timing samples` in component calibration, before a full step was verified. `kaggle kernels output` returned only a zero-byte log file; the traceback text was relayed from the live Kaggle run.
- Root cause: component tuning compared candidates with a baseline cohort but treated baseline timing noise as fatal without a bounded repeat.
- Repair: one exact baseline component remeasurement when and only when baseline estimate fails the relative-MAD stability guard. Keep the full requested beam; replace the unstable cohort only if the retry is stable. If the retry is unstable or the calibration deadline is exhausted, raise a named error.
- Source fix: `01603caf3be3ec176c9360ce2ed4718858e04ee0`.
- Regression: `python -m pytest integrations/cayleypy_native/tests/test_component_autotune.py integrations/cayleypy_native/tests/test_pipeline_autotune.py -q` -> 23 passed.
- Broader package attempt: 315 passed, 8 skipped, 42 failed, and 2 collection errors in this Windows sandbox. Failures are existing platform/environment issues (read-only temp-file rename, installed CayleyPy API mismatch) and unrelated test paths; no claim that the full suite passed.
- Next: publish this narrowly scoped source fix, pin the revised Kaggle setup, rerun only the long 2.5M/2×T4 notebook, and collect its final output plus replay evidence. Do not call the 2.5M long-puzzle run accepted until it completes a full search result and replay.
