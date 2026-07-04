# Stream1 LibTorch Transformer Pretranspose + CUDA Graph Smoke

Date: 2026-07-04

Scope:

- Store LibTorch projection weights as transposed-contiguous KxH tensors at load time instead of transposing HxK on every `at::linear` call.
- Add explicit `--cuda-graph` benchmark mode using long-lived static tensors, non-default CUDA stream, side-stream warmup, and `at::cuda::CUDAGraph` capture/replay.
- Update the Kaggle 2xT4 package to benchmark both `eager` and `cuda_graph` modes in one run and optionally run a short Nsight Systems graph profile if `nsys` is present.
- Keep the default/MLP/native build Torch-free and unchanged.

Local build checks:

```text
cmake -S /workspace -B /tmp/build-libtorch-graph-pretranspose \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake \
  -DBEAM_ENABLE_LIBTORCH_STREAM1=ON \
  -DCUTLASS_DIR=/opt/cutlass \
  -DBEAM_CUDA_ARCHITECTURES=75
cmake --build /tmp/build-libtorch-graph-pretranspose --target stream1_transformer_libtorch_benchmark -j2
```

Result:

```text
[100%] Built target stream1_transformer_libtorch_benchmark
```

Default no-LibTorch guard:

```text
contract_tests=pass
```

Notes:

- This is still a benchmark/scaffold, not production dispatcher wiring.
- Real graph capture/replay behavior must be proven on Kaggle/real GPU by v6 output and profiler evidence.