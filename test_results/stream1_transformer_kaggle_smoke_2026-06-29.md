# Stream1 Transformer Kaggle Smoke Package 2026-06-29

## Scope
- Created package `kaggle_t4_transformer_smoke/` for a private Kaggle 2xT4 smoke run.
- No Kaggle push, GitHub push, or remote kernel execution was performed.
- This note documents local package creation and validation only.

## Package
- `kaggle_t4_transformer_smoke/kernel-metadata.json`
  - id: `trydotatwo/cayley-beam-transformer-2xt4-smoke`
  - private: `true`
  - GPU enabled: `true`
  - machine shape: `NvidiaTeslaT4`
  - internet enabled: `true`
  - model source: `vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1`
- `kaggle_t4_transformer_smoke/t4-transformer-beam-smoke.ipynb`
  - first cell exposes GitHub source, branch, model source, beam/depth/puzzle, torchrun topology, and Stream1/3/4/history parameters.
  - clone/build/export transients stay under `/tmp`.
  - exported transformer weights and logs stay under `/kaggle/working`.
  - `.pth` discovery requires exactly one match under the configured Kaggle model source and prints candidate roots plus all discovered `.pth` files on failure.
  - exporter command uses `tools/export_stream1.py --format piece-transformer --dtype fp16 --num-classes 120`.
  - build targets are `production_runner` and `stream1_transformer_cuda_tests` with `BEAM_CUDA_ARCHITECTURES=75`.
  - smoke launch uses two-rank `torch.distributed.run --no-python`, puzzle `0`, depth `3`, beam `1048576`, `B_MICRO=512`, `STREAM1_CONCURRENCY=1`, `STREAM3_RING_SLOTS=2`, small manual shard settings, and `BEAM_WEIGHT_DIR=/kaggle/working/stream1_transformer_weights_fp16`.
  - run validation requires log evidence of `stream1_backend=piece_transformer` or `stream1_model_backend=piece_transformer`.

## Local Validation
- `py` JSON parse plus Python AST parse passed for all three notebook code cells.
- `nbformat.validate` passed after adding stable notebook cell ids.
- `py -m json.tool kaggle_t4_transformer_smoke/kernel-metadata.json` passed.
- Git diff/status reviewed before staging; pre-existing untracked build artifacts were left untouched.

## Remote Run
- Actual Kaggle execution is intentionally not included in this task and can be appended later by the coordinator.
