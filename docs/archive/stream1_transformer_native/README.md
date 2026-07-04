# Stream1 Native Transformer Archive

Date: 2026-07-04
Branch: codex/stream1-piece-transformer

## Decision

The native CUDA/CUTLASS Stream1 piece-transformer backend is correct and close to the PyTorch fast path, but the best verified 2xT4 result is still not a clear win. For production transformer inference, the next implementation direction is an explicit C++ LibTorch execution backend for the same `piece_transformer` exported weights.

This is not a fallback and not distillation. Model format remains `piece_transformer`; `libtorch` is an execution backend selected explicitly.

## Best Native Evidence

| Run | Commit/tag | Backend notes | Best GPU0 | Best GPU1 | 2xT4 aggregate | Ratio vs PyTorch batch_process aggregate |
|---|---|---:|---:|---:|---:|---:|
| v15 | `stream1-transformer-biasadd256-b34bf7c` | fp16 cols=256 half2 bias-add fast path | 614318.7 | 599458.0 | 1213776.7 | 0.962 |
| v16 | `stream1-transformer-bias-ln-65bf43d` | fused bias + LayerNorm | 623236.4 | 587309.8 | 1210546.2 | 0.960 |
| v18 | `stream1-transformer-block51-optin-7adbe14` | exact p900 block51, opt-in | 610697.3 | 601347.2 | 1212044.5 | 0.961 |
| v19 | `stream1-transformer-extended-sweep-2ac6373` | extended B_MICRO/concurrency sweep, block51 opt-in | 626703.8 | 598031.9 | 1224735.7 | 0.971 |

Reference PyTorch batch_process evidence: 630697.0 candidates/s per T4, aggregate 1261394.0 candidates/s.

## Archived Unverified Experiment

`block51_bias_ln_aborted_2026-07-04.patch` contains the abandoned attempt to make the exact-shape `block51` path optionally fuse projection bias into following LayerNorm via `BEAM_STREAM1_TRANSFORMER_BLOCK51_BIAS_LN=1`.

It was archived before verification and intentionally removed from the working CUDA source. Keep it only as an idea sketch, not as production code.

## Ideas Kept For Later

- specialized C++/CUDA tensor builder for fast piece tokens, reusable by LibTorch via from-blob tensors or a custom extension;
- LibTorch C++ SDPA path using the exported `fast_slot_projected`, `fast_piece_static`, p900 positions/masks, and `eps=1e-5` LayerNorm contract;
- production LibTorch path should be an explicit backend, fail loudly if not built, and should not silently route to native CUDA;
- CUDA Graph integration must be proven separately, because arbitrary Torch op dispatch inside existing dispatcher graph capture is risky;
- if production LibTorch remains faster but not graph-capturable, add a named non-graph Stream1 execution path instead of hiding it behind the native graph path.

## Files To Check

- `tools/stream1_transformer_torch_benchmark.py` is the PyTorch-equivalent forward spec.
- `tools/export_stream1_transformer.py` defines the exported weight contract.
- `tools/stream1_weight_io.hpp` validates the manifest and raw weight shapes.
- `cuda/stream1_transformer.cu` remains the native CUDA/CUTLASS transformer backend.