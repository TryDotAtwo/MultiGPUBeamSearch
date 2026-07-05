# Stream1 Transformer Backend Compare Kaggle v5 2026-07-05

## Run

- kernel=`trydotatwo/cayley-beam-transformer-backend-compare-2xt4` version 5
- checked_out_commit=`1c069c5`
- branch=`codex/stream1-piece-transformer`
- model=`vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1`
- runtime_seconds=`1316.3`
- output_dir=`test_results/kaggle_transformer_backend_compare_v5_2026-07-05/`

Version 4 failed after native eager GPU0 because the Kaggle package parser expected `scratch_bytes` immediately after `candidates_per_sec`. The Stream1 native benchmark now emits `checksum`, `score_key_digest`, and `first_score_keys` before `scratch_bytes`. Version 5 fixed the parser and completed.

## Aggregate Results

| rank | backend/mode | aggregate best candidates/s | vs v3 |
|---:|---|---:|---:|
| 1 | torch_original_batch_process/batch_process | 1421505.5 | 1.0500x |
| 2 | libtorch/eager | 1385290.0 | 1.1109x |
| 3 | libtorch/cuda_graph | 1357923.0 | 1.1201x |
| 4 | native_cutlass/graph | 1130487.0 | 0.9708x |
| 5 | native_cutlass/eager | 1117222.4 | 0.9899x |
| 6 | torch_exported/eager | 982797.1 | 1.0076x |

Previous v3 aggregates used for comparison:

| backend/mode | v3 aggregate candidates/s |
|---|---:|
| torch_original_batch_process/batch_process | 1353772.5 |
| libtorch/eager | 1247021.0 |
| libtorch/cuda_graph | 1212341.0 |
| native_cutlass/graph | 1164436.6 |
| native_cutlass/eager | 1128672.0 |
| torch_exported/eager | 975414.6 |

## Best Per-GPU Rows

| backend/mode/gpu | best candidates/s | config |
|---|---:|---|
| libtorch/eager/gpu0 | 726210.0 | `batch=192,iters=100` |
| torch_original_batch_process/batch_process/gpu1 | 710774.6 | `batch=65536,eval_batch_size=2048,iters=20` |
| torch_original_batch_process/batch_process/gpu0 | 710730.9 | `batch=65536,eval_batch_size=2048,iters=20` |
| libtorch/cuda_graph/gpu0 | 705290.0 | `batch=192,iters=100` |
| libtorch/eager/gpu1 | 659080.0 | `batch=192,iters=100` |
| libtorch/cuda_graph/gpu1 | 652633.0 | `batch=192,iters=100` |
| native_cutlass/graph/gpu0 | 576788.3 | `b_micro=384,concurrency=1` |
| native_cutlass/eager/gpu1 | 566224.8 | `b_micro=512,concurrency=1` |
| native_cutlass/graph/gpu1 | 553698.7 | `b_micro=384,concurrency=1` |
| native_cutlass/eager/gpu0 | 550997.6 | `b_micro=384,concurrency=1` |
| torch_exported/eager/gpu0 | 526509.5 | `batch=768,iters=50` |
| torch_exported/eager/gpu1 | 456287.6 | `batch=320,iters=50` |

## Interpretation

- Original PyTorch `batch_process` remains the fastest overall at `1.4215M` aggregate candidates/s.
- Explicit C++ LibTorch eager is now the fastest exported-weight backend at `1.3853M`, about `0.9745x` of original PyTorch and `1.225x` of native CUTLASS graph in this run.
- C++ LibTorch CUDA Graph works but is slower than eager on Kaggle T4 here: `1.3579M` vs `1.3853M`.
- Native CUTLASS is stable but behind LibTorch on T4 for this transformer shape. Further native work should target layout/LayerNorm/epilogue structure; random fused residual+bias was already rejected.

## Verification Artifacts

- `test_results/kaggle_transformer_backend_compare_v5_2026-07-05/stream1_transformer_backend_compare_summary.json`
- `test_results/kaggle_transformer_backend_compare_v5_2026-07-05/stream1_transformer_backend_compare_rows.csv`
- `test_results/kaggle_transformer_backend_compare_v5_2026-07-05/cayley-beam-transformer-backend-compare-2xt4.log`