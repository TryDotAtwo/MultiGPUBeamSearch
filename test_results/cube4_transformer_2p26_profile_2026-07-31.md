# Cube4 Transformer 2xT4 beam 2^26 profile selection (2026-07-31)

## Workload

- Hardware: real Kaggle 2x Tesla T4, torchrun world size 2.
- Global beam: 67,108,864 (`2**26`); local beam: 33,554,432 per rank.
- Puzzle: Cube4 synthetic puzzle 0; depth limit 8 inclusive.
- Transformer output dimension: 24; `B_MICRO=384`.

## Results

| profile | Stream3 candidates | physical rings | final chunk | depth-8 seconds | peak MiB/GPU | avg util | status |
|---|---:|---:|---:|---:|---:|---:|---|
| v14 control | 18,432 | 2 | coupled/default | 3,304.95 | 13,535 | 99.42/99.95% | stable |
| v16 prior winner | 18,432 | 4 active S4 | coupled/default | 3,157.70 | 13,295 | 99.98/99.86% | stable |
| accumulator 88 slots | 811,008 | 2 | coupled/default | 3,001.40 | 14,587 | 99.67% | stable |
| accumulator 192 + final88k | 1,769,472 | 2 | 88,064 | **2,765.04** | **13,003** | 99.60% | stable |
| accumulator 192 + final98k | 1,769,472 | 2 | 98,304 | 3,302.62 | 13,025 | 99.69/99.70% | stable |

Selected profile: accumulator 192 + final chunk 88,064. It is 12.43% faster than v16 and 16.34% faster than v14. The 98,304 A/B is 19.44% slower than 88,064 despite similar VRAM, so it is rejected.

## Integrated runtime

- `b_micro=384`
- `stream1_concurrency=1`
- `stream3_ring_slots=192`
- `ring_count=2`
- `shard_count=8`
- `shard_capacity_scale_ppm=1_000_000`
- `stream4_batch_candidates=196_608`
- `stream4_trigger_candidates=393_216`
- `stream4_active_sort_slots=1`
- `final_materialize_chunk_candidates=88_064`

## Evidence and gates

- 88k output: `test_results/cube4_transformer_s3final88k_v1_2026-07-31/`
- 98k output: `test_results/cube4_transformer_s3final98k_v1_2026-07-31/`
- Universal public Python suite: `259 passed`.
- Generated notebook: JSON parsed; all seven code/markdown cells structurally valid and all code cells passed Python AST parsing.
- No CUDA/Stream4 algorithm was changed during parameter selection. `tools/production_runner.cu` was synchronized to the already approved/pinned public commit `f312265` so the universal collection tests and runtime match the notebook pin.