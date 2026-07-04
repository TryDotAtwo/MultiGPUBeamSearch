# Stream1 LibTorch Transformer Linear Projection Smoke

Date: 2026-07-04

Scope:

- Replace C++ `torch::matmul(x, weight_hxk) + bias` projection calls with `at::linear(x, weight_hxk.transpose(0, 1), bias)`.
- Apply to QKV, attention output, FF1, FF2, and final output projections.
- Preserve no-QKV-contiguous SDPA path, dense token build, and `c10::InferenceMode`.
- Do not change MLP/default native Stream1 paths.

Local build check:

```text
docker run --rm -v D:\100XH100\.worktrees\stream1-piece-transformer:/workspace -w /workspace cmz-native-dev:2026-05-26 /bin/bash -lc 'cmake -S /workspace -B /tmp/build-libtorch-linear -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake -DBEAM_ENABLE_LIBTORCH_STREAM1=ON -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75 && cmake --build /tmp/build-libtorch-linear --target stream1_transformer_libtorch_benchmark -j2'
```

Result:

```text
[100%] Built target stream1_transformer_libtorch_benchmark
```

Notes:

- This is an execution-backend benchmark experiment to better match PyTorch `linear` dispatch.
- Real speed impact requires Kaggle 2xT4 v5 measurement.