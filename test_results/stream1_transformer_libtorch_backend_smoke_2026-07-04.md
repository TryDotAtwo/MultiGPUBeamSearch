# Stream1 Transformer LibTorch C++ Backend Smoke

Date: 2026-07-04

Scope:

- Add explicit opt-in C++ LibTorch Stream1 piece-transformer benchmark target.
- Preserve existing MLP/CUTLASS/native transformer build by default.
- Do not add fallback or distillation behavior.

Build checks:

```text
cmake -S . -B build-libtorch-stream1 \
  -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') \
  -DBEAM_ENABLE_LIBTORCH_STREAM1=ON \
  -DCUTLASS_DIR=/opt/cutlass \
  -DBEAM_CUDA_ARCHITECTURES=75
cmake --build build-libtorch-stream1 --target stream1_transformer_libtorch_benchmark -j2
```

Result:

```text
[100%] Built target stream1_transformer_libtorch_benchmark
```

CPU forward smoke using `test_results/stream1_transformer_reference/weights_fp16`:

```text
stream1_transformer_libtorch_backend=1 device=cpu dtype=fp16 seq_len=51 d_model=256 nhead=8 layers=4 output_dim=24
stream1_transformer_libtorch_micro batch=2 iters=1 elapsed_ms=322.759 parents_per_sec=6.19657 candidates_per_sec=148.718 checksum=2426304
```

Default no-LibTorch build check in `gpu-dev-cutlass-nsight:cuda128-sm120`:

```text
cmake -S . -B build-no-libtorch-check -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75
cmake --build build-no-libtorch-check --target contract_tests -j2
./build-no-libtorch-check/contract_tests
contract_tests=pass
```

Notes:

- The C++ LibTorch tool uses the same exported `piece_transformer` manifest and raw weight files as the Python Torch benchmark.
- It calls LibTorch `scaled_dot_product_attention` for the attention core.
- This is an execution-backend benchmark/scaffold, not a production dispatcher fallback.
- Production dispatcher integration still needs an explicit graph-capture decision: either prove LibTorch op dispatch capture safety with stable allocations, or add a named non-graph Stream1 path for LibTorch.