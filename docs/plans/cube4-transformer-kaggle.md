# Cube4 Transformer Kaggle Integration Plan

## Global constraints

- Do not change the beam-search algorithm or CUDA search architecture.
- Add only the model-layout/export/runtime metadata support needed for the supplied `cube4` checkpoint.
- Preserve existing `p900` behavior and unrelated dirty changes.
- Use TDD: each production behavior change must be preceded by a focused failing test.
- Keep the Kaggle validation notebook private.
- Validate on two T4 GPUs and retain logs/artifacts under `test_results/`.

## Task 1: Cube4 exporter and runtime contract

Extend the Stream1 piece-transformer export/runtime metadata contract to support the supplied cube4 model:

- `state_len=96`
- `num_classes=6`
- `move_count=output_dim=24`
- `num_pieces=56`
- `max_piece_size=3`
- `num_piece_types=3`
- `seq_len=57` with CLS pooling
- `piece_layout=cube4`
- `piece_embed_mode=piece_local`
- activation `relu`

Export the cube4 piece position/mask/type tables from the trusted model layout and retain exact p900 support. Add focused tests that fail before implementation and pass afterward. Do not touch search semantics.

## Task 2: Private Kaggle 2xT4 validation notebook

Package the supplied model and a thin private Kaggle notebook. The notebook must:

- verify checkpoint and metadata hashes;
- clone the exact integration branch/commit;
- export the checkpoint to Stream1 FP16;
- build and run focused transformer tests;
- use two T4 ranks;
- load standard CayleyPy cube4 competition data;
- run a bounded smoke solve with clear depth/timing/result logs;
- save solution/submission artifacts locally.

## Task 3: Evidence and review

Download Kaggle outputs, verify two-rank execution, exported manifest, depth progress, solution validity, and artifact paths. Record exact kernel version, parameters, timings, and any unresolved limitation in `test_results/`, `memory/CHANGELOG.md`, and `memory/PROMPTS.md`.
