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
## 2026-07-05 Backend Parity Contract

The three Stream1 `piece_transformer` backends now share a parity launcher contract. PyTorch and LibTorch benchmarks emit `checksum` and `first_score_keys`, and the native CUTLASS transformer microbenchmark emits the same fields while optionally using `BEAM_STREAM1_SYNTHETIC_STATES=1` for deterministic synthetic inputs.

`tools/stream1_transformer_parity.py` is the correctness gate for comparing explicit backends on one synthetic batch. It supports a dry-run mode for clean checkout validation and an execution mode for GPU hosts with exported weights and built backend binaries. This is only a verification layer; production dispatch still selects one backend explicitly and must not silently fall back.
## 2026-07-05 Score-Key Digest Gate

The backend parity contract now includes `score_key_digest`, a deterministic FNV-1a 64-bit digest over all quantized score keys in batch-major, move-major order. The digest is emitted by the PyTorch benchmark, the C++ LibTorch benchmark, and the native CUTLASS transformer microbenchmark. The parity runner requires the digest to exist and reports exact digest agreement, but default cross-backend pass/fail uses first-row score-key tolerance because different FP16 projection kernels can shift quantized keys slightly. Use `--require-exact-digest` for strict bit-exact comparisons inside one backend family.
## 2026-07-05 Backend Mode Parity Update

`tools/stream1_transformer_parity.py` now accepts explicit backend modes in `backend:mode` form, so one run can compare `pytorch`, `libtorch:eager`, `libtorch:cuda_graph`, `native_cutlass:eager`, and `native_cutlass:graph`. The parser now only consumes true result rows, avoiding the previous `_report=` line collision in PyTorch logs.

The C++ LibTorch FFN activation now calls `at::silu(...)` directly after the FF1 `at::linear` projection, matching the Python reference expression at the operator level while keeping the opt-in LibTorch target separate from default MLP/native builds.
## 2026-07-05 LibTorch T4 Candidate Update

Kaggle 2xT4 LibTorch-only sweeps v8-v10 identified explicit C++ LibTorch eager as the current fastest exported-weight transformer path. The best v10 row was `batch=384` on both T4s, aggregate `1534314.0` candidates/s, about `1.079x` of the full backend compare v5 original PyTorch `batch_process` reference (`1421505.5`). CUDA Graph capture remains available as an explicit mode, but it was slower than eager in the v10 run.

The next production integration should therefore be a named non-graph Stream1 LibTorch eager path, not an attempt to force LibTorch into the existing dispatcher CUDA graph templates. The path must remain explicit, fail closed when LibTorch is unavailable, and keep the MLP/default native paths Torch-free.
## 2026-07-05 LibTorch Repeated-Pass Stability

Kaggle 2xT4 LibTorch-only benchmark v11 repeated each batch three times to reduce single-row noise. The stable candidate remains eager LibTorch at `batch=384` on both T4s: aggregate mean `1467441.7` candidates/s, best aggregate `1497577.0`, and `1.032x` of the full backend compare v5 original PyTorch `batch_process` aggregate. CUDA Graph is still explicit and functional, but slower in this benchmark: aggregate mean `1411365.0`, best aggregate `1442113.0`, and graph/eager best `0.963x`.

Production wiring should use the repeated-pass result as the default LibTorch sizing point and keep CUDA Graph as a benchmark mode until a graph-capture path beats eager under the same measurement contract.
## 2026-07-05 Dispatcher Ring-Slot Hook

The dispatcher now has an optional `DispatcherRingSlotLauncher` hook at the ring-slot scheduling boundary. The default path is unchanged: when no hook is supplied, MLP and native `piece_transformer` Stream1 still launch the pre-instantiated ring-slot CUDA Graph templates. The hook is intentionally Torch-free and only exposes the ring/slot/job context, CUDA streams, parent window, and score-ring offset. This gives a named non-graph Stream1 execution path a stable integration point without linking LibTorch into the default `beam_cuda` build.

A test passthrough launcher exercises the hook by launching the existing graph exec and checking that the hook call count matches `ring_slot_jobs_launched`. This is scaffolding for explicit `libtorch:eager` production integration, not a fallback route.
## 2026-07-05 LibTorch Production Runner Hook

The production runner now has an explicit opt-in C++ LibTorch Stream1 path for exported `piece_transformer` weights. The default `production_runner` target remains Torch-free and still uses the existing pre-instantiated ring-slot CUDA Graph templates for MLP and native CUTLASS Stream1. The new `production_runner_libtorch_stream1` target is built only with `BEAM_ENABLE_LIBTORCH_STREAM1=ON` and is selected at runtime with `BEAM_STREAM1_EXECUTOR=libtorch_eager`.

The LibTorch production path uses the dispatcher ring-slot hook as a named non-graph Stream1 executor. It skips native Stream1 weight upload/scratch and native ring-slot graph instantiation, writes quantized score keys into the existing score ring, then launches the existing Stream2 hash/goal kernel on the paired Stream2 lane. Stream3/Stream4/Stream5/finalization remain the normal production pipeline.

This is not a fallback. If the binary is not built with LibTorch support, if the selected model is not `backend=piece_transformer`, if `BEAM_STREAM1_MODE=uniform` is requested, or if `output_dim != MOVE_COUNT`, the runner fails closed.
