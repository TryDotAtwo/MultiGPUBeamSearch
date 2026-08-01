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
- `BEAM_WIDTH=262144`
- `DEPTH_LIMIT=82`
- `SHARD_COUNT=1`
- `SHARD_CAPACITY_SCALE_PPM=2000000`
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


Version 2 reached depth 4 with `BEAM_WIDTH=65536`, `SHARD_COUNT=1`, then failed in Stream3 because the small beam left no spill reserve: `spill_capacity=0`, `shard_capacity_candidates=34816`. Version 3 increases only the beam to `262144` while keeping `SHARD_COUNT=1`; the comparison target remains puzzle `991` reference length `82`.


Version 3 reached depth 31 with `BEAM_WIDTH=262144`, `SHARD_COUNT=1`, then failed in Stream3 because shard capacity was still tight: `existing=138167`, `raw_count=7563`, `shard_capacity_candidates=138240`, `spill_capacity=0`. Version 4 keeps `BEAM_WIDTH=262144` and raises `SHARD_CAPACITY_SCALE_PPM` to `2000000`.


Version 4 completed successfully:

- kernel status: `KernelWorkerStatus.COMPLETE`
- `puzzle_solved=1`
- `puzzle_id=991`
- `seconds=2463.89`
- `solution_length=82`
- reference example length for puzzle `991`: `82`
- comparison: match
- output directory: `test_results/kaggle_t4_transformer_solve_v4_output/`
- torchrun log: `test_results/kaggle_t4_transformer_solve_v4_output/stream1_transformer_solve_logs/torchrun_piece_transformer_solve_p991_d82_b262144.log`

Solution:

`DR.R.-BR.-B.BL.-B.-DL.-DL.D.DL.DR.-FL.-R.-R.-FL.D.F.FL.DR.B.-DL.-L.-F.D.U.-BR.B.-BL.F.-L.F.FR.R.FR.-DR.-R.-DL.-DR.-R.L.FL.L.-FL.-F.-FR.F.F.FR.F.R.-FR.-F.L.F.L.DR.FL.R.DR.DL.DR.-B.-DR.B.B.D.-L.-BL.BR.B.-BR.BL.L.-DR.-B.DL.-BL.-B.DR.BL.-DL.-L`

Speed notes from the successful run:

- The run used `BEAM_WIDTH=262144`, `SHARD_COUNT=1`, `SHARD_CAPACITY_SCALE_PPM=2000000`, `STREAM3_RING_SLOTS=2`, and `B_MICRO=512`.
- Late full-beam depths were about `33s/depth` on 2xT4; example tail shows depth 76 at `33.1978s` and depth 77 at `33.1503s`.
- The reference PyTorch notebook reports puzzle `991` length 82 in `209.78s`, but it is a single-phase model-guided sampler with `B=65536`, not the full multi-stream beam pipeline with reconstruction/history and global thresholding.
