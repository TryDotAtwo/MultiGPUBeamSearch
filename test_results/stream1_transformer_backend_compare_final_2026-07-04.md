# Stream1 Transformer Backend Compare Final 2026-07-04

## Local RTX 3070 Laptop

| rank | backend | best candidates/s | config |
|---:|---|---:|---|
| 1 | native_cutlass_eager | 1078358.9 | `{'b_micro': 256, 'concurrency': 1, 'rows_per_launch_group': 256, 'ms_per_iter': 5.6975}` |
| 2 | native_cutlass_graph_targeted | 1059577.3 | `{'b_micro': 256, 'concurrency': 1, 'rows_per_launch_group': 256, 'ms_per_iter': 5.7985}` |
| 3 | libtorch_eager | 823300.0 | `{'batch': 1536, 'elapsed_ms': 2238.79}` |
| 4 | libtorch_cuda_graph_targeted | 692990.0 | `{'batch': 1024, 'iters': 20, 'elapsed_ms': 709.275}` |
| 5 | torch_exported | 692372.5 | `{'batch': 4096, 'iters': 50, 'ms_per_iter': 141.9813671875, 'elapsed_ms': 7099.068359375}` |

Local note: original Kaggle PyTorch `batch_process` was not available locally; `torch_exported` is the exported-weight PyTorch path.

## Kaggle 2xT4

- kernel=trydotatwo/cayley-beam-transformer-backend-compare-2xt4 v3
- checked_out_commit=1b6a9a7
- runtime_seconds=1362.7

| rank | backend/mode | aggregate best candidates/s |
|---:|---|---:|
| 1 | torch_original_batch_process/batch_process | 1353772.5 |
| 2 | libtorch/eager | 1247021.0 |
| 3 | libtorch/cuda_graph | 1212341.0 |
| 4 | native_cutlass/graph | 1164436.6 |
| 5 | native_cutlass/eager | 1128672.0 |
| 6 | torch_exported/eager | 975414.6 |

| backend/mode/gpu | best candidates/s | config |
|---|---:|---|
| torch_original_batch_process/batch_process/gpu1 | 679390.1 | `{'batch': 65536, 'eval_batch_size': 2048, 'iters': 20}` |
| torch_original_batch_process/batch_process/gpu0 | 674382.4 | `{'batch': 65536, 'eval_batch_size': 2048, 'iters': 20}` |
| libtorch/eager/gpu0 | 659410.0 | `{'batch': 384, 'iters': 100}` |
| libtorch/cuda_graph/gpu0 | 636773.0 | `{'batch': 384, 'iters': 100}` |
| libtorch/eager/gpu1 | 587611.0 | `{'batch': 192, 'iters': 100}` |
| native_cutlass/graph/gpu1 | 582897.4 | `{'batch': 256, 'b_micro': 256, 'concurrency': 1, 'rows_per_launch_group': 256}` |
| native_cutlass/graph/gpu0 | 581539.2 | `{'batch': 768, 'b_micro': 256, 'concurrency': 3, 'rows_per_launch_group': 768}` |
| libtorch/cuda_graph/gpu1 | 575568.0 | `{'batch': 192, 'iters': 100}` |
| native_cutlass/eager/gpu1 | 568343.3 | `{'batch': 512, 'b_micro': 256, 'concurrency': 2, 'rows_per_launch_group': 512}` |
| native_cutlass/eager/gpu0 | 560328.7 | `{'batch': 384, 'b_micro': 384, 'concurrency': 1, 'rows_per_launch_group': 384}` |
| torch_exported/eager/gpu0 | 520278.0 | `{'batch': 448, 'iters': 50}` |
| torch_exported/eager/gpu1 | 455136.6 | `{'batch': 192, 'iters': 50}` |

Conclusion:
- Local winner: native CUTLASS eager.
- Kaggle 2xT4 winner: original PyTorch `batch_process`.
- Kaggle explicit C++ LibTorch eager is the fastest exported-weight path in this run, ahead of native CUTLASS.
