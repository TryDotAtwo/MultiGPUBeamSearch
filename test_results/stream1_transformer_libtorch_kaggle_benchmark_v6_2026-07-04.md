# Stream1 LibTorch Transformer Kaggle 2xT4 Benchmark v6

Date: 2026-07-04

Branch: `codex/stream1-piece-transformer`
Commit: `66d43ca`
Kernel: `trydotatwo/cayley-beam-transformer-libtorch-2xt4-benchmark`, Kaggle version 6
Model source: `vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1`

Change under test:

- Load all projection weights once as transposed-contiguous KxH tensors for direct `at::linear(x, weight_kxh, bias)`.
- Add explicit C++ LibTorch `--cuda-graph` benchmark mode using:
  - long-lived static input/output tensors,
  - non-default CUDA stream from `c10::cuda::getStreamFromPool`,
  - side-stream warmup,
  - `at::cuda::CUDAGraph::capture_begin/capture_end/replay`.
- Kaggle package benchmarks both `eager` and `cuda_graph` in one run.
- Kaggle package probes Nsight Systems; Kaggle image returned `NSYS_PROFILE_UNAVAILABLE=nsys_not_found`.

Result:

| Mode | GPU0 best | GPU1 best | Aggregate | vs PyTorch ref | vs v5 eager |
|---|---:|---:|---:|---:|---:|
| eager | 680638.0 | 684619.0 | 1365257.0 | 1.0823x | 1.0231x |
| cuda_graph | 650796.0 | 665838.0 | 1316634.0 | 1.0438x | 0.9867x |

Graph over eager aggregate: `0.9644x`.

Conclusion:

CUDA Graph capture/replay works for the C++ LibTorch Stream1 transformer path on Kaggle 2xT4. This answers the technical graph feasibility question positively for the benchmark scaffold.

However, graph replay is slower than eager for the working batch range in this microbenchmark. The likely reason is that this workload is dominated by GPU kernel time and library kernels, while CUDAGraph may freeze graph-safe library execution choices or lose some eager dispatch behavior. Without Nsight Systems on the Kaggle image, this cannot be diagnosed at timeline level there.

The pretransposed KxH weights are a speed win for eager mode: v6 eager reached `1365257.0`, improving over v5 `1334475.0` by about `1.023x`.

Production implication:

- Keep the graph-capable code path, but do not assume it is faster yet.
- Before production routing, profile graph vs eager on a machine with Nsight Systems installed.
- Production graph path must remain explicit and fail-closed: fixed batch/shape/device/dtype, static tensors, warmup before capture, no dynamic fallback.

Artifacts:

- `test_results/kaggle_libtorch_transformer_benchmark_v6_2026-07-04/stream1_libtorch_transformer_summary.json`
- `test_results/kaggle_libtorch_transformer_benchmark_v6_2026-07-04/stream1_libtorch_transformer_rows.csv`
- `test_results/kaggle_libtorch_transformer_benchmark_v6_2026-07-04/cayley-beam-transformer-libtorch-2xt4-benchmark.log`