# Stream1 LibTorch Narrow State Slice

Date: 2026-07-05

Scope:

- Continue explicit C++ LibTorch Stream1 `piece_transformer` performance work.
- Keep `pytorch`, `libtorch`, and `native_cutlass` as explicit backends.
- Do not add fallback or distillation behavior.
- Do not touch the MLP/default Torch-free production path.

## Change Kept

`PieceTransformerLibTorch::build_tokens()` now slices the logical state bytes with `narrow(1, 0, state_len)` instead of tensor indexing with `Slice()`.

This is a small LibTorch token-build/layout cleanup. It preserves the same tensor shape and semantics, but avoids the more generic indexing path. The benchmark startup line reports `state_slice=narrow` so Kaggle logs show which implementation ran.

## Rejected Experiment

Tried replacing:

```cpp
pieces = pieces + gathered * slot_mask_values[slot];
```

with in-place `pieces.addcmul_(gathered, slot_mask_values[slot])`.

Result: Python unit tests and small Docker parity passed, but real exported weights in CUDA Graph mode failed with a CUDA device-side assert in ATen `indexSelectLargeIndex`. This experiment was reverted and is not part of the kept code.

## Local Verification

Python gates:

```text
python tests\test_stream1_transformer_backends.py
python tests\test_stream1_transformer_parity.py
```

Result:

```text
5 tests passed
7 tests passed
```

Docker CUDA real-weight parity:

```text
stream1_transformer_parity_status=pass
backends=pytorch:eager,libtorch:eager,libtorch:cuda_graph
device=cuda:0
batch=192
tolerance=3072
weights=test_results/kaggle_libtorch_transformer_benchmark_v1_2026-07-04/stream1_transformer_weights_fp16
```

Local RTX 3070 Laptop A/B was noisy, but `narrow` passed both eager and graph runs:

| run | mode | batch | mean candidates/s | best candidates/s |
|---|---|---:|---:|---:|
| baseline | eager | 192 | 625893.0 | 650208.0 |
| baseline | eager | 384 | 466743.3 | 547581.0 |
| baseline | cuda_graph | 192 | 485460.0 | 577427.0 |
| baseline | cuda_graph | 384 | 471140.7 | 502692.0 |
| narrow | eager | 192 | 540226.7 | 629976.0 |
| narrow | eager | 384 | 692020.7 | 750674.0 |
| narrow | cuda_graph | 192 | 623188.0 | 690956.0 |
| narrow | cuda_graph | 384 | 586848.7 | 596020.0 |

## Kaggle 2xT4 Verification

Private Kaggle LibTorch-only stability run v12 completed on commit `7c25b2e`.

Artifacts:

- `test_results/kaggle_libtorch_transformer_stability_v12_2026-07-05/stream1_libtorch_transformer_summary.json`
- `test_results/kaggle_libtorch_transformer_stability_v12_2026-07-05/stream1_libtorch_transformer_rows.csv`
- `test_results/kaggle_libtorch_transformer_stability_v12_2026-07-05/cayley-beam-transformer-libtorch-2xt4-benchmark.log`

Best aggregate rows:

| mode | GPU0 best | GPU1 best | aggregate best | aggregate mean-best |
|---|---:|---:|---:|---:|
| eager | 755015.0 | 780120.0 | 1535135.0 | 1501530.7 |
| cuda_graph | 710771.0 | 748435.0 | 1459206.0 | 1416463.0 |

Comparison against v11 repeated-pass stability:

| mode | v11 best aggregate | v12 best aggregate | ratio | v11 mean aggregate | v12 mean aggregate | ratio |
|---|---:|---:|---:|---:|---:|---:|
| eager | 1497577.0 | 1535135.0 | 1.0251 | 1467441.7 | 1501530.7 | 1.0232 |
| cuda_graph | 1442113.0 | 1459206.0 | 1.0119 | 1411365.0 | 1416463.0 | 1.0036 |

Reference original PyTorch `batch_process` aggregate from backend compare v5: `1421505.5` candidates/s. v12 eager is `1.0799x` by best aggregate and `1.0563x` by mean-best aggregate versus that reference.

## Decision

Keep the `narrow` state-slice patch. It is a small verified LibTorch improvement on 2xT4 and does not affect default Torch-free MLP/native builds.

CUDA Graph remains an explicit benchmark mode, not the production default, because repeated T4 runs still show eager faster than graph.