# Stream1 MLP Native vs PyTorch Kaggle 2xT4 - 2026-07-01

Kaggle kernel: `trydotatwo/cayley-beam-mlp-native-vs-torch-2xt4-benchmark`, version 2.

Source tag: `stream1-mlp-torch-kaggle-v2-code` at `cc90a3d`.

Status: `KernelWorkerStatus.COMPLETE`.

## Best Per-GPU Throughput

| GPU | Native B_MICRO | Native concurrency | Native candidates/s | Torch rows | Torch candidates/s | Native / Torch |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4096 | 1 | 16,416,335.8 | 16,384 | 8,133,621.3 | 2.0183x |
| 1 | 4096 | 1 | 15,728,371.2 | 4,096 | 6,763,818.6 | 2.3254x |

Aggregate best throughput across both T4 GPUs:

- Native CUTLASS MLP: 32,144,707.0 candidates/s.
- PyTorch preindexed MLP: 14,897,439.9 candidates/s.
- Native / Torch aggregate: 2.1577x.

## Evidence Files

Downloaded Kaggle outputs are under:

- `test_results/kaggle_t4_mlp_benchmark_v2_output/mlp_native_vs_torch_comparison.csv`
- `test_results/kaggle_t4_mlp_benchmark_v2_output/native_mlp_benchmark_rows.csv`
- `test_results/kaggle_t4_mlp_benchmark_v2_output/torch_mlp_benchmark_rows.csv`
- `test_results/kaggle_t4_mlp_benchmark_v2_output/cayley-beam-mlp-native-vs-torch-2xt4-benchmark.log`

## Interpretation

For the exported folded MLP weights, the native CUTLASS Stream1 path is about 2.0-2.3x faster than a direct PyTorch implementation on Kaggle T4. This establishes the local comparison baseline for judging the transformer backend: the transformer PyTorch fast path being faster than native means the transformer native path still has implementation overhead, not that native CUDA is inherently slower than Torch for this project.