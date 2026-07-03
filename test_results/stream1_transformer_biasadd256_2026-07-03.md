# Stream1 Transformer Bias Add FP16 256-Wide Fast Path

Date: 2026-07-03
Branch: `codex/stream1-piece-transformer`

## Change

Added a transformer-local fp16 `cols == 256` `bias_add` launch path in
`cuda/stream1_transformer.cu`. The specialized kernel uses one block per row and
128 threads, with each thread processing one `half2`. BF16 and non-256 shapes
continue through the existing generic dtype path.

This is not a fallback path and does not touch the MLP Stream1 runtime.

## Verification

Docker image: `gpu-dev-cutlass-nsight:cuda128-sm120`

```text
stream1_transformer_cuda_tests=pass
dispatcher_cuda_tests=pass
contract_tests=pass
```

Local graph benchmark on the current development GPU:

```text
BEAM_STREAM1_TRANSFORMER_B_MICRO=512
BEAM_STREAM1_TRANSFORMER_CONCURRENCY=2
candidates_per_sec=638354.3
scratch_bytes=327729152
```

Nsight Systems graph profile, `B_MICRO=1024`, `CONCURRENCY=1`:

```text
stream1_transformer_bias_add_kernel before LN128+biasadd256: 7.0%, avg 289347 ns
stream1_transformer_bias_add256_fp16_kernel after:          3.7%, avg 151671 ns
profile benchmark candidates_per_sec=723336.6
```

## Interpretation

The dedicated fp16 256-wide path roughly halves the remaining standalone
`bias_add` kernel cost in the local graph profile. End-to-end T4 impact still
needs a Kaggle run because local RTX laptop measurements are noisy and not a
T4/A100 performance gate.