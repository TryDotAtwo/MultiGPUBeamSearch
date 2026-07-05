# Stream1 Transformer Static Plan And Scratch Accounting 2026-07-05

## Scope

- Apply the useful low-level inference ideas that are safe for the current explicit backend architecture:
  - LibTorch: precompute per-slot position/table/mask tensors for token construction and report native-equivalent activation bytes.
  - Native CUTLASS: centralize transformer scratch sizing into a shared byte plan and expose token/QKV/attention/context/FF/logit byte breakdown in benchmark output.
  - Compatibility: accept legacy manifests where `move_count` is absent and equals `output_dim`.
- Preserve explicit backend selection only: `pytorch`, `libtorch`, and `native_cutlass`; no fallback or distillation behavior.
- Preserve MLP/default Torch-free build behavior.

## Local Verification

Source commit tested locally before Kaggle: `53ddb28`.

Build/checks completed in Docker:

```text
native sm75 build: stream_benchmark + contract_tests built in gpu-dev-cutlass-nsight:cuda128-sm120
opt-in LibTorch sm75 build: stream1_transformer_libtorch_benchmark + production_runner_libtorch_stream1 built in cmz-native-dev:2026-05-26
default no-LibTorch gate: contract_tests=pass
local sm86 build: stream_benchmark + stream1_transformer_libtorch_benchmark built in cmz-native-dev:2026-05-26
python backend registry tests: 5/5 pass
python parity parser tests: 7/7 pass
kaggle package JSON/AST validation: pass
```

Five-mode local parity on RTX 3070 Laptop GPU passed:

```text
out_dir=test_results/stream1_transformer_plan_local_sm86_2026-07-05_r3
status=pass
backends=pytorch:eager,libtorch:eager,libtorch:cuda_graph,native_cutlass:eager,native_cutlass:graph
score_key_digest=79446185743582083 for all five modes
first_score_keys=all zero for the synthetic fixture
```

Local sm86 speed from the same parity run:

| backend/mode | candidates/s |
|---|---:|
| native_cutlass/graph | 1013226.5 |
| native_cutlass/eager | 998170.0 |
| libtorch/cuda_graph | 836011.0 |
| libtorch/eager | 811433.0 |
| pytorch/eager | 603399.8 |

The native benchmark now reports the same scratch plan used by runtime allocation. For batch 256 it reported:

```text
scratch_bytes=81932288
token_bytes=6684672
qkv_bytes=20054016
attention_bytes=21757952
context_bytes=6684672
ff_hidden_bytes=26738688
logits_bytes=12288
```

## Kaggle 2xT4 Verification

Kernel:

```text
trydotatwo/cayley-beam-transformer-backend-compare-2xt4
version=6
status=COMPLETE
checked_out_commit=53ddb28
runtime_seconds=1385.1
model_source=vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1
exported_manifest=piece_transformer fp16 seq_len=51 d_model=256 nhead=8 layers=4 output_dim=24
```

Downloaded artifacts:

```text
test_results/kaggle_transformer_backend_compare_v6_2026-07-05/
```

Aggregate best rows:

| rank | backend/mode | aggregate candidates/s |
|---:|---|---:|
| 1 | libtorch/eager | 1332366.0 |
| 2 | torch_original_batch_process/batch_process | 1315226.0 |
| 3 | libtorch/cuda_graph | 1303242.0 |
| 4 | native_cutlass/eager | 1125348.9 |
| 5 | native_cutlass/graph | 1086181.4 |
| 6 | torch_exported/eager | 954223.2 |

Best per-GPU rows:

| backend/mode/gpu | candidates/s | config |
|---|---:|---|
| torch_original_batch_process/batch_process/gpu1 | 731055.5 | batch=65536 eval_batch_size=2048 |
| libtorch/eager/gpu0 | 681230.0 | batch=192 iters=100 |
| libtorch/cuda_graph/gpu0 | 652303.0 | batch=128 iters=100 |
| libtorch/eager/gpu1 | 651136.0 | batch=192 iters=100 |
| libtorch/cuda_graph/gpu1 | 650939.0 | batch=192 iters=100 |
| native_cutlass/graph/gpu1 | 585467.3 | b_micro=384 concurrency=2 |
| native_cutlass/eager/gpu1 | 585416.6 | b_micro=512 concurrency=1 |
| torch_original_batch_process/batch_process/gpu0 | 584170.5 | batch=65536 eval_batch_size=2048 |
| native_cutlass/eager/gpu0 | 539932.3 | b_micro=256 concurrency=1 |
| native_cutlass/graph/gpu0 | 500714.1 | b_micro=256 concurrency=1 |

## Interpretation

- Correctness gate passed locally across all explicit backend modes.
- Kaggle full compare completed successfully on the current pushed source.
- LibTorch eager remains the best exported-weight path in this full compare and slightly beat the original notebook `batch_process` in this run (`1.013x`).
- The static LibTorch token plan is not a proven T4 speed win: v6 is lower than the best previous full-compare v5 LibTorch aggregate (`1.332M` vs `1.385M`) and lower than focused repeated-pass v11 (`1.467M` mean). Treat it as neutral/uncertain, not as the next speed breakthrough.
- The native scratch byte plan is useful and low-risk: it removes duplicate sizing formulas and makes runtime/benchmark memory accounting auditable by phase.
- CUDA Graph remains functional for LibTorch but is slower than eager on T4 in this compare.

## Follow-up

- Keep `libtorch/eager` as the current production candidate for the explicit transformer backend.
- Use the new scratch breakdown to guide memory budget decisions before any larger production solve.
- The next real speed work should target either PyTorch/LibTorch token-build/layout overhead with a measured repeated-pass gate, or native CUTLASS LayerNorm/epilogue layout. Random static-token caching alone is not enough.