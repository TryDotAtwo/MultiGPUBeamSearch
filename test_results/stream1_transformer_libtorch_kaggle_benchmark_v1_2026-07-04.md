# Stream1 Transformer LibTorch Kaggle 2xT4 Benchmark v1

Date: 2026-07-04

Kernel: `trydotatwo/cayley-beam-transformer-libtorch-2xt4-benchmark`, version 1
Status: `KernelWorkerStatus.COMPLETE`
Source branch: `codex/stream1-piece-transformer`
Source commit checked by notebook: `d3525a4`
Model source: `vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1`
Exported runtime weights: `piece_transformer`, `fp16`, `seq_len=51`, `d_model=256`, `nhead=8`, `layers=4`, `output_dim=24`

Command path:

- Kaggle mounted the `.pth` model artifact.
- The script exported it with `tools/export_stream1.py --format piece-transformer --dtype fp16`.
- The script built `stream1_transformer_libtorch_benchmark` with `BEAM_ENABLE_LIBTORCH_STREAM1=ON`.
- The script benchmarked GPU0 and GPU1 separately with `CUDA_VISIBLE_DEVICES=<gpu>` and summed best per-GPU throughput.

Best rows:

```text
GPU0 batch=384 iters=100 elapsed_ms=1959.90 parents_per_sec=19592.9 candidates_per_sec=470229.0 checksum=465862656
GPU1 batch=384 iters=100 elapsed_ms=1930.93 parents_per_sec=19886.8 candidates_per_sec=477283.0 checksum=465862656
2xT4 aggregate candidates_per_sec=947512.0
```

Comparison:

```text
PyTorch batch_process reference per T4: 630697.0 candidates/s
PyTorch batch_process 2xT4 aggregate: 1261394.0 candidates/s
C++ LibTorch benchmark aggregate / PyTorch aggregate: 0.7512x
Native/CUTLASS v19 aggregate: 1224735.7 candidates/s
C++ LibTorch benchmark aggregate / native v19 aggregate: 0.7736x
```

Conclusion:

The current C++ LibTorch backend is a correct opt-in execution scaffold, but this direct eager LibTorch benchmark is slower than both the PyTorch notebook fast path and the best native/CUTLASS path. The likely causes are C++ eager op dispatch and allocation/layout overhead in the current wrapper path; it is not yet using a production Stream1 graph/non-graph integration strategy.

Artifacts:

- `test_results/kaggle_libtorch_transformer_benchmark_v1_2026-07-04/stream1_libtorch_transformer_summary.json`
- `test_results/kaggle_libtorch_transformer_benchmark_v1_2026-07-04/stream1_libtorch_transformer_rows.csv`
- `test_results/kaggle_libtorch_transformer_benchmark_v1_2026-07-04/stream1_libtorch_transformer_logs/stream1_libtorch_transformer_gpu0.log`
- `test_results/kaggle_libtorch_transformer_benchmark_v1_2026-07-04/stream1_libtorch_transformer_logs/stream1_libtorch_transformer_gpu1.log`
- `test_results/kaggle_libtorch_transformer_benchmark_v1_2026-07-04/cayley-beam-transformer-libtorch-2xt4-benchmark.log`

Note: Kaggle output download also included the temporary exported weight directory. The local copy is ignored by Git and is about 6.5 MB, but Windows refused deletion with `Access is denied`. The benchmark package was updated after v1 so future runs call `cleanup_path(WEIGHT_OUT_DIR)` before kernel completion and should not publish the temporary weights as outputs.