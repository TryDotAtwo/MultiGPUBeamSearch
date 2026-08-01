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

## Remote Kaggle Run V1
- Kernel: `trydotatwo/cayley-beam-transformer-2xt4-smoke`, version `1`.
- Status: `KernelWorkerStatus.COMPLETE`.
- GitHub branch cloned by notebook: `codex/stream1-piece-transformer`.
- Output directory: `test_results/kaggle_t4_transformer_smoke_v1_output/`.
- Selected checkpoint: `/kaggle/input/models/vladkuznetsov266/megaminx-qtransformer-1782210824/pytorch/default/1/megaminx-transformer/weights/p900-t000-q-rw-sym_1782210824_best.pth`.
- Export manifest: `backend=piece_transformer`, `dtype=fp16`, `seq_len=51`, `d_model=256`, `layers=4`, `output_dim=24`.
- Torchrun smoke: puzzle `0`, depth `3`, beam `1048576`, `TORCHRUN_NPROC_PER_NODE=2`, `B_MICRO=512`, `STREAM1_CONCURRENCY=1`, `STREAM3_RING_SLOTS=2`.
- Runtime evidence: rank log contains `stream1_model_backend=piece_transformer`, `stream1_backend=piece_transformer`, and `stream1_transformer_dims seq_len=51 ... output_dim=24`.
- Result: `RUN_T4_TRANSFORMER_SMOKE_RC 0 seconds=4.603`; final solver line `puzzle_solved=0 puzzle_id=0 seconds=0.713001 solution_length=-1 solution=`. Depth-3 smoke was not expected to solve puzzle 0.
- Note: `stream1_transformer_cuda_tests` built but skipped on Kaggle because the ignored local reference fixture is intentionally absent in a clean GitHub checkout. The production 2-rank transformer runner path did execute successfully.
