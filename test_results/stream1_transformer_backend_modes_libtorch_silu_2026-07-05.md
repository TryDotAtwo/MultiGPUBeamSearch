# Stream1 Transformer Backend Modes and LibTorch SiLU

Date: 2026-07-05
Branch: `codex/stream1-piece-transformer`

## Scope

- Keep three explicit Stream1 `piece_transformer` backends available: `pytorch`, `libtorch`, and `native_cutlass`.
- Extend the parity runner to compare explicit backend modes such as `libtorch:cuda_graph` and `native_cutlass:graph`.
- Keep default/MLP builds Torch-free; no fallback and no distillation behavior.
- Try one native CUTLASS epilogue optimization, keep it only if it improves the relevant workload.

## Changes

- `tools/stream1_transformer_parity.py` now parses `backend:mode` entries in `--backends`; a bare backend defaults to `eager`.
- The parity parser now matches only result rows with `<prefix> `, so `torch_stream1_transformer_report=...` no longer overwrites the real PyTorch result row.
- Real runs still require every backend to emit `score_key_digest`, but exact digest equality is optional through `--require-exact-digest`. Default cross-backend pass/fail uses first-row score-key tolerance and reports exact digest agreement separately.
- The C++ LibTorch FFN activation now uses `at::silu(at::linear(...))` instead of spelling SiLU as `x * sigmoid(x)`.

## Real Five-Mode Parity

Command shape:

```text
python tools/stream1_transformer_parity.py \
  --weight-dir test_results/kaggle_libtorch_transformer_benchmark_v1_2026-07-04/stream1_transformer_weights_fp16 \
  --build-dir build-real-parity \
  --device cuda:0 \
  --batch 256 \
  --iters 3 \
  --backends pytorch,libtorch,libtorch:cuda_graph,native_cutlass,native_cutlass:graph \
  --out-dir test_results/stream1_transformer_libtorch_silu_parity_2026-07-05
```

Result:

```text
stream1_transformer_parity_status=pass
```

Key rows from `test_results/stream1_transformer_libtorch_silu_parity_2026-07-05/stream1_transformer_parity.json`:

| backend | mode | checksum | score_key_digest | max first-row diff vs PyTorch |
|---|---|---:|---:|---:|
| pytorch | eager | 310566912 | 3945694622387745667 | 0 |
| libtorch | eager | 310624256 | 3621528375435670403 | 32 |
| libtorch | cuda_graph | 310624256 | 3621528375435670403 | 32 |
| native_cutlass | eager | 310636288 | 7767622256714949507 | 44 |
| native_cutlass | graph | 310636288 | 7767622256714949507 | 44 |

The exact digest differs across backend families because the FP16 projection kernels/order differ, but the score-key row deltas are well inside the current `3072` tolerance.

## LibTorch SiLU Speed Smoke

Longer local RTX 3070 Laptop micro rows after the `at::silu` change:

```text
eager,256,20,174.679,29310.8,703460,cuda:0,fp16
cuda_graph,256,20,166.082,30828.1,739874,cuda:0,fp16
```

These rows are from:

- `test_results/stream1_transformer_libtorch_silu_eager_2026-07-05.csv`
- `test_results/stream1_transformer_libtorch_silu_graph_2026-07-05.csv`

## Rejected Native Epilogue Experiment

A native CUTLASS residual+bias epilogue experiment compiled and passed parity, and improved the small `b256,c1` point, but it regressed the more relevant `b512,c2` native workload to `849196.8` candidates/s. The CUDA source was reverted; only the evidence remains:

- `test_results/stream1_transformer_residual_bias_epilogue_parity_2026-07-05/`
- `test_results/stream1_transformer_residual_bias_epilogue_b512_c2_2026-07-05.md`

## Final Verification

Local Python checks:

```text
ast_ok=1
python tests\test_stream1_transformer_parity.py       -> Ran 7 tests, OK
python tests\test_stream1_transformer_backends.py     -> Ran 5 tests, OK
PYTHONPATH=. python tests\test_stream1_transformer_exporter.py -> Ran 6 tests, OK
```

Docker no-LibTorch guard in `gpu-dev-cutlass-nsight:cuda128-sm120`:

```text
contract_tests=pass
no_libtorch_guard=pass
```

Docker opt-in LibTorch build in `cmz-native-dev:2026-05-26`:

```text
libtorch_target_build=pass
```

## Status

- Kept: parity parser/mode support and LibTorch `at::silu`.
- Reverted: native CUTLASS residual+bias epilogue experiment due benchmark regression.
- Verified: default MLP/no-LibTorch path still builds and tests without Torch; opt-in LibTorch target builds separately.