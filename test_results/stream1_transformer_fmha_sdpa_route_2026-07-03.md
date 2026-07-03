# Stream1 Piece Transformer CUTLASS FMHA SDPA Route

## Scope

Implement the first PyTorch-like attention route for Stream1 `piece_transformer`: use a single CUTLASS example-41 fused multi-head attention kernel for the Q/K/V attention stage instead of the older decomposed `QK GemmBatched -> softmax -> PV GemmBatched` chain.

This is transformer-only. The existing MLP Stream1 path is not touched.

## Code Changes

- `cuda/stream1_transformer.cu`
  - selects CUTLASS FMHA for fp16 on SM75/T4 and fp16/bf16 on SM80+.
  - removes the legacy tensor-attention backend from attention dispatch.
  - fails closed when CUTLASS FMHA headers are unavailable; no CUDA fallback path is selected.
- `cuda/stream1_transformer_fmha.cu`
  - adds SM75 fp16 instantiation of the CUTLASS FMHA kernel.
  - keeps SM80 bf16 and fp16 instantiations.
- `cuda/stream1_transformer_fmha.hpp`
  - passes the explicit SM75-fp16 selection flag.

## Verification

Docker image: `gpu-dev-cutlass-nsight:cuda128-sm120`

Compile checks:

```text
cmake --build build-stream1-transformer-fmha-sm75 --target stream1_transformer_cuda_tests stream_benchmark -j2
cmake --build build-stream1-transformer-fmha-sm86 --target stream1_transformer_cuda_tests dispatcher_cuda_tests contract_tests stream_benchmark -j2
```

Both SM75 and SM86 targets built successfully. CUTLASS header warnings were emitted, but there were no project compile errors.

Runtime checks:

```text
stream1_transformer_cuda_tests=pass
dispatcher_cuda_tests=pass
contract_tests=pass
```

Whitespace check:

```text
git diff --check: pass
```

## Local Benchmark Notes

A single local SM86 graph-benchmark point was run after the code change:

```text
BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1
BEAM_STREAM1_TRANSFORMER_B_MICRO=1024
BEAM_STREAM1_TRANSFORMER_CONCURRENCY=4
candidates_per_sec=613097.9
scratch_bytes=1310916608
```

This local timing should not be treated as the T4 decision point because another GPU container was active on the machine during the measurement. Earlier local sweep evidence in this same change set reached higher graph points, but the actual gate for this route is a fresh Kaggle 2xT4 benchmark.

## Interpretation

This completes the first step requested by the user: make native attention structurally closer to PyTorch SDPA, using one fused attention kernel rather than the older decomposed native chain. The next improvement target is reducing remaining transformer overhead around QKV/FMHA layout packing and the non-attention GEMM/LayerNorm epilogues, then validating on Kaggle T4.