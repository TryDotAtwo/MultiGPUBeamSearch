# Stream1 Transformer Fused Bias + LayerNorm

Date: 2026-07-04
Branch: `codex/stream1-piece-transformer`

## Change

Fused transformer projection bias with the following LayerNorm step in
`cuda/stream1_transformer.cu`.

- Attention-output bias is applied inside the same kernel that computes LN2.
- FF2 bias is delayed until the next block's LN1, where it is applied and stored
  back to the residual token buffer before that block uses the residual.
- Final-block FF2 bias is applied directly inside the CLS output LayerNorm, since
  only the CLS vector is consumed by the classifier after the final transformer
  block.

This removes the standalone transformer bias-add kernels from the main forward
loop without changing the MLP Stream1 runtime.

## Verification

Docker image: `gpu-dev-cutlass-nsight:cuda128-sm120`

SM86/FMHA build and tests:

```text
stream1_transformer_cuda_tests=pass
dispatcher_cuda_tests=pass
contract_tests=pass
```

SM75/T4-oriented build and test:

```text
stream1_transformer_cuda_tests=pass
```

Local graph benchmark on the current development GPU:

```text
BEAM_STREAM1_TRANSFORMER_B_MICRO=512
BEAM_STREAM1_TRANSFORMER_CONCURRENCY=2
candidates_per_sec=750833.0
scratch_bytes=327729152
```

Local Nsight graph profile, `B_MICRO=1024`, `CONCURRENCY=1`:

```text
profile benchmark candidates_per_sec=775386.2
stream1_transformer_bias_layernorm256_copy_kernel: 49 calls, 10.8%, avg 468486.8 ns
stream1_transformer_layernorm256_copy_kernel:      14 calls,  2.7%, avg 402704.8 ns
standalone stream1_transformer_bias_add kernel:    absent from the graph
```

For comparison, the previous biasadd256 profile reported `723336.6`
candidates/s at `1024x1` and still had standalone bias-add kernels at `3.7%` of
kernel time.

## Interpretation

The fused bias+LayerNorm path removes eight standalone bias-add graph nodes per
transformer inference and improves local graph replay by about `1.072x` at
`1024x1` compared with the previous biasadd256 profile. The remaining hot path is
still dominated by CUTLASS GEMM/fused-epilogue kernels plus attention; this patch
mainly reduces launch/node count and memory traffic around normalization.