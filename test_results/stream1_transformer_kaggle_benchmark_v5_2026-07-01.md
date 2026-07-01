# Stream1 Transformer Kaggle 2xT4 Benchmark v5 - 2026-07-01

Kernel: `trydotatwo/cayley-beam-transformer-2xt4-benchmark`
Version: 5
Status: `KernelWorkerStatus.COMPLETE`
Branch commit tested: `78052c5` (`Clean Stream1 transformer tensor attention`)

The notebook now fails fast if the GitHub checkout is not commit `78052c5`.
The downloaded Kaggle log confirms:

```text
GITHUB_COMMIT= 78052c5
```

## Best Rows

| GPU | B_MICRO | concurrency | rows/group | ms/group | parents/s | candidates/s | scratch bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 512 | 2 | 1024 | 60.5191 | 16920.3 | 406086.6 | 327729152 |
| 1 | 512 | 2 | 1024 | 59.2023 | 17296.6 | 415118.8 | 327729152 |

## Aggregate

| version | aggregate parents/s | aggregate candidates/s |
|---|---:|---:|
| v4 tensor GEMM attention | 33884.5 | 813226.3 |
| v5 clean tensor attention | 34216.9 | 821205.4 |

v5 is only `1.01x` v4 on Kaggle 2xT4. The local clean-path improvement did not
transfer to Kaggle T4 throughput.

## Comparison To PyTorch Notebook Fast Path

The public `vladkuznetsov266/transformer-inference-example` fast PyTorch path
measured on Kaggle T4:

- Direct model forward best: `734786.8` candidates/s at `batch=2048`.
- Searcher-like `batch_process`: `630697.0` candidates/s for `65536` rows with
  `eval_batch_size=16384`.

Per T4, the current native CUDA Stream1 piece-transformer backend is:

- GPU0: `406086.6 / 734786.8 = 0.55x` of direct PyTorch forward.
- GPU1: `415118.8 / 734786.8 = 0.57x` of direct PyTorch forward.
- GPU0: `406086.6 / 630697.0 = 0.64x` of PyTorch `batch_process`.
- GPU1: `415118.8 / 630697.0 = 0.66x` of PyTorch `batch_process`.

So yes: on Kaggle T4, the native backend is still slower than the PyTorch fast
path per GPU, despite running correctly and using the intended commit.

## Interpretation

This run proves the clean tensor attention rewrite did not fix the Kaggle T4
performance gap. The best point stayed at `B_MICRO=512, concurrency=2` on both
T4s, with roughly the same scratch footprint as v4.

The next optimization should not assume the local Docker microbenchmark result
represents Kaggle T4. The Kaggle evidence points back to profiling the exact T4
runtime and reducing transformer-only launch/elementwise overhead around the
linear GEMMs, LayerNorms, residual adds, bias+activation, and QKV/V layout path.

## Artifacts

Downloaded output directory:
`test_results/kaggle_t4_transformer_benchmark_v5_output/`

Useful files:

- `stream1_transformer_benchmark_rows.csv`
- `cayley-beam-transformer-2xt4-benchmark.log`
- `stream1_transformer_benchmark_logs/stream1_transformer_benchmark_gpu0.log`
- `stream1_transformer_benchmark_logs/stream1_transformer_benchmark_gpu1.log`
- `stream1_transformer_benchmark_reports/stream1_transformer_benchmark_gpu0.md`
- `stream1_transformer_benchmark_reports/stream1_transformer_benchmark_gpu1.md`
