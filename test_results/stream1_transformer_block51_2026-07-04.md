# Stream1 Transformer Block51 Specialized Backend

Date: 2026-07-04
Branch: `codex/stream1-piece-transformer`

## Change

Added a shape-specialized `block51` Stream1 piece-transformer path for the current
p900 transformer shape:

```text
state_len=120
num_classes=120
num_pieces=50
max_piece_size=3
seq_len=51
d_model=256
nhead=8
head_dim=32
transformer_layers=4
ff_dim=1024
output_dim=24
```

The top-level transformer entry now routes this exact shape through
`stream1_transformer_inference_block51_cuda`. Non-matching shapes remain on the
existing checked generic path.

The first block51 specialization fuses input token construction with input
LayerNorm in `stream1_transformer_build_input_layernorm51x256_kernel`, then uses
the measured-better T4 `biasadd256` block schedule rather than the v16 fused
bias+LayerNorm schedule.

This is a transformer-only path and does not touch the MLP Stream1 runtime.

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
candidates_per_sec=753966.2
scratch_bytes=327729152
```

Local Nsight graph profile, `B_MICRO=1024`, `CONCURRENCY=1`:

```text
profile benchmark candidates_per_sec=749701.0
stream1_transformer_build_input_layernorm51x256_kernel: 7 calls, 2.2%, avg 687731.0 ns
stream1_transformer_layernorm256_copy_kernel:          56 calls, 10.6%, avg 413442.1 ns
stream1_transformer_bias_add256_fp16_kernel:           56 calls, 4.0%, avg 154706.9 ns
CUTLASS GEMM/fused-epilogue groups:                    about 73.6% combined
attention_kernel_batched_impl:                         9.6%
```

## Interpretation

This is the first real block-specialized foundation. It removes the separate
input-build graph stage and hard-routes p900 through an exact-shape schedule, so
future `seq/d/head/ff` variants can add their own guarded schedule instead of
mutating the generic transformer path.

The remaining wall time is still dominated by GEMM/epilogue kernels, so the next
block51-specific speed work should target one of:

- specialized LN+QKV projection for `rows = b_micro * 51`, `K=256`, `N=768`;
- SM75/SM80 tile selection for QKV, FF1, FF2, and residual projections;
- a custom block-local residual+bias epilogue that avoids the slower broadcast
  GEMM path observed in prior experiments.