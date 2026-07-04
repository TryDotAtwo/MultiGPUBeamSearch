# Stream1 Transformer Kaggle 2xT4 v19 Extended Sweep

Date: 2026-07-04

Source:

- tag: `stream1-transformer-extended-sweep-2ac6373`
- commit: `2ac6373`
- benchmark mode: `BEAM_STREAM1_TRANSFORMER_BLOCK51=1`
- sweep: `B_MICRO={256,384,512,768,1024,1536,2048,3072,4096}`, `concurrency={1,2,3,4,5,6}`

Best rows:

```text
GPU0: b_micro=512 concurrency=2 candidates_per_sec=626703.8 scratch_bytes=327729152
GPU1: b_micro=384 concurrency=1 candidates_per_sec=598031.9 scratch_bytes=122898432
2xT4 aggregate candidates_per_sec=1224735.7
```

Comparison:

```text
PyTorch batch_process reference per T4: 630697.0 candidates/sec
PyTorch batch_process 2xT4 aggregate: 1261394.0 candidates/sec
Native v19 aggregate / PyTorch aggregate: 0.9709
```

Conclusion:

The extended native CUDA/CUTLASS sweep is close to PyTorch but still not a clear win. This result supports moving the production transformer track to an explicit C++ LibTorch execution backend while archiving native experiments for later reference.

Artifacts:

- raw Kaggle log: `test_results/kaggle_stream1_transformer_2xt4_v19_extended_sweep_2026-07-04/cayley-beam-transformer-2xt4-benchmark.log`
- parsed rows: `test_results/kaggle_stream1_transformer_2xt4_v19_extended_sweep_2026-07-04/stream1_transformer_benchmark_rows.csv`