# Stream1 Piece Transformer Strided-QKV FMHA

## Scope

Try the next improvement after the PyTorch-like FMHA route: remove the explicit QKV repack kernel and pass the original QKV projection buffer directly to CUTLASS FMHA with strided Q/K/V pointers.

This keeps the transformer-only backend boundary and does not touch the MLP Stream1 path.

## Code Changes

- `cuda/stream1_transformer_fmha.cu`
  - `query_ptr = qkv`
  - `key_ptr = qkv + 256`
  - `value_ptr = qkv + 512`
  - `q/k/v_strideM = 768`
  - `q/k/v_strideH = 32`
  - `q/k/v_strideB = 51 * 768`
  - removed the dead QKV pack kernel/helper from the file.

## Verification

Docker image: `gpu-dev-cutlass-nsight:cuda128-sm120`

Compile checks:

```text
cmake --build build-stream1-transformer-fmha-sm75 --target stream1_transformer_cuda_tests stream_benchmark -j2
cmake --build build-stream1-transformer-fmha-sm86 --target stream1_transformer_cuda_tests dispatcher_cuda_tests contract_tests stream_benchmark -j2
```

Runtime checks:

```text
stream1_transformer_cuda_tests=pass
dispatcher_cuda_tests=pass
contract_tests=pass
```

## Local Graph Benchmark

Local SM86, graph benchmark mode, using the downloaded v11 fp16 transformer export:

```text
b_micro=512 concurrency=2 candidates_per_sec=698530.6 scratch_bytes=327729152
b_micro=1024 concurrency=1 candidates_per_sec=737089.2 scratch_bytes=327729152
```

These local numbers are not a replacement for T4 validation, but they confirm that direct strided QKV is supported by the CUTLASS FMHA kernel and is faster than the packed-QKV route on the local GPU.

## Next Gate

Run a fresh Kaggle 2xT4 benchmark pinned to the strided-QKV commit and compare against v11 FMHA and the PyTorch reference.