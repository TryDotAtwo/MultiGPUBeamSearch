# Public CayleyPy 2xT4 Kaggle release — 2026-07-30

- Public kernel: `trydotatwo/cayleypy-2xt4-checkpoint-beam-search`
- URL: https://www.kaggle.com/code/trydotatwo/cayleypy-2xt4-checkpoint-beam-search
- Release source commit: `63caece`
- Pinned solver commit: `bb505484a839d3b78819f86aa28e76b842faab09`
- Metadata: public, internet enabled, NvidiaTeslaT4.
- Public result endpoint: `https://cayleypy-results-ingest-staging.tupa-expert.workers.dev`
- Unconfigured template behavior: clean `SETUP_REQUIRED` handoff; no solve attempted.
- Configured contract: standard CayleyPy inputs, checkpoint-only BatchNorm-folded or ResMLP-LayerNorm, fp16, output dimension 1 or move_count, exactly two T4 GPUs.

## Local release gates

- `python -m pytest tests/cayleypy_public -q`: 238 passed.
- Generated notebook JSON and all code-cell ASTs: valid (6 cells).
- `git diff --check`: clean.
- Public package secret scan: clean.

## Remote publication

- Kaggle CLI push receipt: kernel version 1 successfully pushed.
- Initial observed status: `KernelWorkerStatus.RUNNING`.
- Final status is appended after terminal monitoring.
## Terminal verification

- Final Kaggle status: `KernelWorkerStatus.COMPLETE`.
- Downloaded log: `test_results/kaggle_public_cayleypy_release_v1/cayleypy-2xt4-checkpoint-beam-search.log`.
- Log evidence: `SETUP_REQUIRED`, `preflight_skipped`, `run_skipped`, no traceback, notebook conversion completed.

