# Stream1 LibTorch Transformer Fast Token Build Smoke

Date: 2026-07-04

Scope:

- Keep Stream1 MLP/default native builds Torch-free.
- Keep the LibTorch piece-transformer path explicit and opt-in behind `BEAM_ENABLE_LIBTORCH_STREAM1=ON`.
- Try a C++ LibTorch token-build fast path that precomputes active piece indices/positions per slot instead of gathering all 50 pieces and multiplying by a mask every slot.
- Replace `torch::NoGradGuard` with `c10::InferenceMode` in the benchmark process.
- Expand the Kaggle 2xT4 LibTorch sweep around the previous best batch point and make the commit-prefix guard optional for iterative branch benchmarking.

Local build checks:

```text
docker run --rm -v D:\100XH100\.worktrees\stream1-piece-transformer:/workspace -w /workspace cmz-native-dev:2026-05-26 /bin/bash -lc 'cmake -S /workspace -B /tmp/build-libtorch-fastpath -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake -DBEAM_ENABLE_LIBTORCH_STREAM1=ON -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75 && cmake --build /tmp/build-libtorch-fastpath --target stream1_transformer_libtorch_benchmark -j2'
```

Result:

```text
[100%] Built target stream1_transformer_libtorch_benchmark
```

Default no-LibTorch guard:

```text
docker run --rm -v D:\100XH100\.worktrees\stream1-piece-transformer:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 /bin/bash -lc 'cmake -S /workspace -B /tmp/build-no-libtorch-fastpath -DCMAKE_BUILD_TYPE=Release -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75 && cmake --build /tmp/build-no-libtorch-fastpath --target contract_tests -j2 && /tmp/build-no-libtorch-fastpath/contract_tests'
```

Result:

```text
contract_tests=pass
```

Notes:

- No production dispatcher behavior changed in this patch.
- No fallback or distillation behavior was added.
- Local CPU forward smoke was skipped because temporary transformer weights had been removed from the worktree by request; the Kaggle benchmark exports weights at runtime and cleans them before kernel completion.
- Real 2xT4 speed is measured by the follow-up Kaggle v2 benchmark package.