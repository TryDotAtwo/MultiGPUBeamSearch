# Stream1 LibTorch Transformer 2xT4 Fine/Upper Sweep 2026-07-05

## Scope

- Kaggle kernel: `trydotatwo/cayley-beam-transformer-libtorch-2xt4-benchmark`
- Versions: 8, 9, 10
- Repo branch cloned inside Kaggle: `codex/stream1-piece-transformer`
- Checked-out backend commit in v9/v10: `dfd3664`
- Model source: `vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1`
- Backend under test: explicit C++ LibTorch Stream1 `piece_transformer` benchmark, fp16, T4/sm75
- No MLP/default runtime path changes.

## Results

| run | sweep purpose | eager aggregate candidates/s | cuda_graph aggregate candidates/s | graph/eager | best eager config |
|---|---|---:|---:|---:|---|
| v8 | first fine sweep around small batches | 1441134.0 | 1392194.0 | 0.9660x | gpu0 batch=96, gpu1 batch=128 |
| v9 | smaller 64..320 sweep with updated PyTorch reference | 1521158.0 | 1512552.0 | 0.9943x | gpu0 batch=320, gpu1 batch=320 |
| v10 | upper sweep through 768 | 1534314.0 | 1493994.0 | 0.9737x | gpu0 batch=384, gpu1 batch=384 |

Reference for ratio: full backend compare v5 original PyTorch `batch_process` aggregate `1421505.5` candidates/s.

Best v10 LibTorch eager: `1534314.0` aggregate candidates/s, `1.0794x` of the v5 original PyTorch reference.
Best v10 per-GPU rows:

| mode | gpu | batch | candidates/s | elapsed_ms |
|---|---:|---:|---:|---:|
| eager | 0 | 384 | 770271.0 | 1196.46 |
| eager | 1 | 384 | 764043.0 | 1206.21 |
| cuda_graph | 0 | 384 | 749904.0 | 1228.96 |
| cuda_graph | 1 | 80 | 744090.0 | 258.033 |

## Interpretation

- LibTorch eager is currently the fastest verified exported-weight transformer path on Kaggle 2xT4.
- The useful T4 micro-batch is higher than the earlier v5 full-compare winner; v10 found `batch=384` on both GPUs.
- Explicit LibTorch CUDA Graph capture works, but it is not faster than eager in the current Kaggle T4 runs. Keep it as an explicit benchmark mode, not the default production assumption.
- The benchmark package now uses the current v5 PyTorch reference (`1421505.5`) instead of the stale `1261394.0` reference and writes `checked_out_commit` into the summary JSON.

## Artifacts

- `test_results/kaggle_libtorch_transformer_fine_sweep_v8_2026-07-05/`
- `test_results/kaggle_libtorch_transformer_fine_sweep_v9_2026-07-05/`
- `test_results/kaggle_libtorch_transformer_upper_sweep_v10_2026-07-05/`