# Stream1 Transformer Kaggle 2xT4 Benchmark v3 2026-06-30

Kernel: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`
Version: 3
Status: `KernelWorkerStatus.COMPLETE`
Branch commit tested: `a290524` (`Optimize Stream1 transformer attention`)

## Best Rows

| Version | GPU | B_MICRO | concurrency | rows/group | ms/group | parents/s | candidates/s | scratch bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 baseline | 0 | 512 | 1 | 512 | 108.0227 | 4739.7 | 113753.8 | 162963456 |
| v2 baseline | 1 | 512 | 1 | 512 | 103.2894 | 4956.9 | 118966.7 | 162963456 |
| v3 warp attention | 0 | 1024 | 1 | 1024 | 103.9292 | 9852.9 | 236468.7 | 240697344 |
| v3 warp attention | 1 | 1024 | 1 | 1024 | 105.9266 | 9667.1 | 232009.6 | 240697344 |

## Aggregate

| Version | aggregate parents/s | aggregate candidates/s |
|---|---:|---:|
| v2 baseline | 9696.6 | 232720.5 |
| v3 warp attention | 19520.0 | 468478.3 |

## Interpretation

The warp-specialized attention patch roughly doubled Kaggle 2xT4 Stream1 throughput (`468478.3 / 232720.5 = 2.01x`) and removed the float global attention score/probability scratch from the transformer scratch estimate. The best T4 point moved from `B_MICRO=512, concurrency=1` to `B_MICRO=1024, concurrency=1`.

The backend remains much slower than the existing MLP path. The earlier interpretation that the model is intrinsically far beyond the `~5x` envelope is not reliable: the PyTorch fast path for this checkpoint empirically suggests that `~5x` is a plausible target order. The current native backend should therefore be treated as under-optimized, especially around kernel launch count, LayerNorm/residual/activation fusion, CUTLASS GEMM setup overhead, and the gap versus PyTorch `scaled_dot_product_attention` / `torch.compile` fast inference.

## Artifacts

Downloaded output directory: `test_results/kaggle_t4_transformer_benchmark_v3_output/`

Useful files:

- `stream1_transformer_benchmark_rows.csv`
- `stream1_transformer_benchmark_logs/stream1_transformer_benchmark_gpu0.log`
- `stream1_transformer_benchmark_logs/stream1_transformer_benchmark_gpu1.log`
- `cayley-beam-transformer-2xt4-benchmark.log`