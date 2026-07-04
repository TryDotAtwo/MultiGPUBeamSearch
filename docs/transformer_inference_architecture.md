# Transformer Inference Architecture Notes

Date: 2026-06-30
Branch: `codex/stream1-piece-transformer`

## Direction

User direction is explicit: no distillation into MLP and no fallback paths. This work keeps the Stream1 `piece_transformer` backend as a real transformer backend and optimizes it directly.

## Current Model

The current Kaggle p900 `piece_transformer` backend is correct but not yet optimized like the existing MLP backend.

Current manifest shape:

- `state_len=120`
- `num_pieces=50`
- `seq_len=51` including CLS
- `d_model=256`
- `nhead=8`
- `head_dim=32`
- `transformer_layers=4`
- `ff_dim=1024`
- `output_dim=24`
- CLS pooling
- SiLU FFN

One parent state becomes 51 token rows. Each layer runs LayerNorm, QKV projection, attention, attention output projection, another LayerNorm, FFN up/down projections, SiLU, residual adds, then final CLS LayerNorm and output projection.

## Current Backend Optimization Path

The immediate route is to make this exact transformer backend production-grade:

1. Add a transformer Stream1 microbenchmark.
2. Tune transformer-specific `B_MICRO`, concurrency, ring slots, and shard sizing separately from MLP.
3. Remove redundant transformer scratch clears.
4. Remove global `attention_scores_probs` scratch and compute softmax/value accumulation in shared/register storage per `(row, head, query)`.
5. Fuse or combine simple epilogues where safe: bias, SiLU, residual add, score quantization.
6. Keep all steady-state allocations preallocated and graph-capturable.

This preserves the trained model and avoids fallback behavior.

## Smaller Transformer Variants

A smaller transformer such as 2 layers, `d_model=128..192`, and `ff_dim=512..768` could reduce GEMM/scratch cost while staying a transformer backend. It requires relaxing current exporter/runtime shape checks, which currently reject non-4-layer/non-1024-FF models.

Risk: medium quality loss. Runtime friendliness: medium. This is valid only as an explicit transformer backend variant, not a fallback to MLP.

## Piece-Mixer As Transformer-Like Backend

A piece-mixer keeps fixed piece tokens but replaces content-dependent QKV attention with deterministic token mixing and channel mixing:

```text
piece embedding -> channel projection -> repeated [token-mix over pieces, channel-mix over channels] -> pool -> 24 scores
```

This is not an MLP fallback. It would be a new explicit piece-token backend with transformer-like token processing but no softmax attention. It is mostly GEMM plus fused bias/activation/residual kernels and removes attention score/probability scratch.

Risk: medium. Runtime friendliness: high. Engineering cost: medium/high because it needs a new manifest/export/runtime path.

## Transformer Reranking

A transformer-only reranking stage could be considered later, but not as a fallback. It would need explicit Stream1 semantics: score all required candidates or define a formally safe two-stage scoring contract before Stream2/3/4 consume score keys.

This is invasive and should not be the first optimization.

## Recommendation

Do current-backend optimization first: benchmark, remove redundant clears, remove global attention scratch, tune transformer-specific batching/concurrency, and then consider smaller transformer or piece-mixer variants. Do not train or wire an MLP distillation path for this track.
## 2026-07-04 Backend Direction Update

After the native CUDA/CUTLASS v19 extended 2xT4 sweep, the best verified native aggregate is `1224735.7` candidates/s, about `0.971x` of the PyTorch `batch_process` reference aggregate. The native experiments remain archived under `docs/archive/stream1_transformer_native/`.

The transformer direction is now three explicit execution backends for the same `piece_transformer` model family: `pytorch`, `libtorch`, and `native_cutlass`. This does not change the manifest model backend, does not distill to MLP, and does not add fallback behavior. Default builds remain Torch-free; the C++ LibTorch benchmark/tool target is enabled only with `BEAM_ENABLE_LIBTORCH_STREAM1=ON`.

Production dispatcher integration remains a separate step for non-native paths because the current dispatcher captures Stream1+Stream2 CUDA Graph templates. LibTorch op dispatch must either be proven graph-capture safe with stable allocations or run through a named non-graph Stream1 execution path. The PyTorch path stays a Python/Kaggle execution backend unless an explicit bridge is designed.
## 2026-07-04 CUTLASS Profiling Update

A later Docker Nsight pass compared the explicit C++ LibTorch/cuBLAS path against the native CUTLASS Stream1 transformer path on local RTX 3070. The native CUTLASS backend is already the practical cuBLAS replacement route for production: local b384/c1 reached `871235.6` candidates/s and b512/c2 reached `991974.7`, versus local LibTorch eager `578741`. Nsight Compute 2025.1.1 confirmed both cuBLASLt and CUTLASS sampled GEMMs use HMMA tensor cores.

The current bottleneck is not missing tensor-core use. Native CUTLASS sampled kernels are around `40-43%` SM throughput and `62-70%` DRAM throughput, so the next work should tune transformer-specific GEMM/epilogue/layout/LayerNorm structure. A wider `128x128x32` QKV/FF1 tile regressed versus the current `128x64x32` tile and was reverted. Evidence: `test_results/stream1_transformer_cutlass_vs_libtorch_profile_2026-07-04.md`.
## 2026-07-04 Three-Backend Policy

The backend registry is `tools/stream1_transformer_backends.py`; details are in `docs/stream1_transformer_backends.md`.

The three supported backend names are:

- `pytorch`: Python/Torch execution, including the Kaggle notebook path and exported-weight Torch benchmark.
- `libtorch`: explicit C++ LibTorch execution over exported weights.
- `native_cutlass`: native C++/CUDA/CUTLASS execution through the existing Stream1 transformer runtime.

The selection is explicit. None of these paths may silently fall back to another path. They should be optimized and verified independently.
