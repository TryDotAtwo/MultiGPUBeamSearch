# Stream1 Piece-Transformer Kaggle Solve Check

Date: 2026-06-30
Branch: `codex/stream1-piece-transformer`

## Reference Notebook

Downloaded source/output from Kaggle kernel `vladkuznetsov266/transformer-inference-example`.

Reference output command in notebook log:

`single_phase_submission.py --group_id 900 --target_id 0 --model_id 1782210824 --B 65536 --num_attempts 2 --num_steps 200 --eval_batch_size 16384 --start 991 --count 1001 --gpu_ids 0 --inference_backend auto`

Reference first row selected for comparison:

- puzzle: `991`
- reported length: `82`
- model artifact: `vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1`

Reference output files:

- `test_results/kaggle_transformer_inference_example_output/submission.csv`
- `test_results/kaggle_transformer_inference_example_output/transformer-inference-example.log`

## Our Kaggle Package

Package: `kaggle_t4_transformer_solve/`
Kernel id: `trydotatwo/cayley-beam-transformer-2xt4-solve-991`

Config:

- `START_PUZZLE_ID=991`
- `PUZZLE_COUNT=1`
- `BEAM_WIDTH=65536`
- `DEPTH_LIMIT=82`
- `SHARD_COUNT=1`
- `STREAM4_BATCH_CANDIDATES=32768`
- `STREAM4_TRIGGER_CANDIDATES=32768`
- `TORCHRUN_NPROC_PER_NODE=2`
- `CUDA_ARCHITECTURES=75`
- `BEAM_WEIGHT_DIR=/kaggle/working/stream1_transformer_weights_fp16`
- transformer model source unchanged from the reference artifact

Local package validation before Kaggle push:

- `kernel-metadata.json`: JSON parse passed
- `t4-transformer-beam-solve.ipynb`: JSON parse passed
- notebook code cells: Python AST parse passed

## Kaggle Result

Version 1 failed in preflight because the initial solve package inherited the smoke SHARD_COUNT=4: stream3_batch=24576 exceeded shard_capacity=9216 at BEAM_WIDTH=65536. The solve package was corrected to SHARD_COUNT=1 and STREAM4_TRIGGER_CANDIDATES=32768, preserving B_MICRO=512 and STREAM3_RING_SLOTS=2.
