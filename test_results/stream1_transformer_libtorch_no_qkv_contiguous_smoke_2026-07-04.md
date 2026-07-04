# Stream1 LibTorch Transformer No QKV Contiguous Smoke

Date: 2026-07-04

Scope:

- Remove explicit `.contiguous()` copies from Q/K/V tensors before LibTorch `scaled_dot_product_attention`.
- Preserve dense token build, `c10::InferenceMode`, and the explicit opt-in LibTorch benchmark boundary.
- Do not change MLP/default native Stream1 paths.

Local build check:

```text
docker run --rm -v D:\100XH100\.worktrees\stream1-piece-transformer:/workspace -w /workspace cmz-native-dev:2026-05-26 /bin/bash -lc 'cmake -S /workspace -B /tmp/build-libtorch-nocontig -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake -DBEAM_ENABLE_LIBTORCH_STREAM1=ON -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75 && cmake --build /tmp/build-libtorch-nocontig --target stream1_transformer_libtorch_benchmark -j2'
```

Result:

```text
[100%] Built target stream1_transformer_libtorch_benchmark
```

Notes:

- Real performance impact requires Kaggle 2xT4 v4 measurement.
- If SDPA internally materializes copies or chooses a slower path for non-contiguous Q/K/V, this experiment should be reverted.