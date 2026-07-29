# Task 5 private Kaggle real-2xT4 acceptance gate

Date: 2026-07-29

## Outcome

Private Kaggle kernel `trydotatwo/cayleypy-public-task-5-2xt4-gate`, version 2, completed successfully and passed the independent downloaded-evidence validator. The accepted run used exactly two Tesla T4 GPUs, the tracked output-dimension-24 fp16 weights, beam 65,536, depth limit 8, puzzle 1, and zero K1/K2 radii. It ran first mode once and collect mode twice with fresh run/history directories.

Version 1 was a diagnostic-only run. Kaggle normalized the title to a slug containing `task-5`, while the initial embedded metadata omitted that hyphen. Its runtime was otherwise healthy, but the independent provenance validator correctly rejected it. Version 2 embeds and attests the actual private slug and is the sole acceptance result.

## Package and provenance

- Private kernel version: 2 (`COMPLETE`; latest run UTC `2026-07-29T00:36:10.427Z`).
- Generated notebook SHA-256: `9ec24174ca48838e3215f8c159eb4db5bdf1884bf5c13902fa27f98e6d1e0c6f`.
- Kernel metadata SHA-256: `b9a22006ac26a28127cf0f1f1c162acf41350369ac0159bc0ffc5def97b9bc32`.
- Public base `origin/main`: `6f95bd6bdb32b5f6ef7cca32b96967bce6036503`.
- Reviewed source commit: `6830401ed2086921d2563c2bc3c11faf6c5a0741`.
- Embedded/overlaid `production_runner.cu`: 242,054 bytes, SHA-256 `f7d20a2fdec5748052b09804a2b2878cb13f854b8dd29e05db92d6828c223774`.
- CUTLASS checkout: `afa1772203677c5118fcd82537a9c8fefbcc7008`.
- Release SM75 binary: 6,100,856 bytes, SHA-256 `c86919b8994ef38f735e6b6159c68198530767bf42a22d17d9bd907c30d6a0ac`.
- Build attestation: `Release`, CUDA architecture `75`, NCCL dynamically linked.
- Build times: base clone 4.146392 s, CUTLASS clone 2.871753 s, configure 4.507312 s, compile 64.314610 s.
- Raw build log SHA-256: `4bbf3ab10ca642cc5ea18833feb2f829f04a2ec929bef52d257c43cd892bf334`.
- Raw source manifest SHA-256: `9264cee58c071e83ab9bf30cb451516ecad86d44571ac3575013ec3f154fa9f2`.
- Raw gate summary SHA-256: `43903d4fa0e6f8fff7eb7da881207c537ffdcd16b1423dc8f3a1dd54f878afa7`.

## Hardware, model, and exact run contract

- GPUs: `0, Tesla T4, 15360 MiB, 14912 MiB free`; `1, Tesla T4, 15360 MiB, 14912 MiB free`.
- CUDA compiler: 12.8 (`V12.8.93`); PyTorch `2.10.0+cu128`; CMake `3.31.10`; Ninja `1.13.0`.
- Model contract: state length 120, 120 classes, `output_dim=24`, `dtype=fp16`; every tracked weight file was size/SHA-attested in `weights_manifest.json` without the private `source_weights` field.
- `torch.distributed.run`: one node, two processes, `--redirects=3`, `--tee=0:3`, native `production_runner` (`--no-python`).
- Puzzle 1; beam 65,536; depth limit 8; maximum unique solutions 16; solved-result capacity 786,432; K1=0; K2=0.
- Measured output-move-count p16 profile: local beam 32,768; B_MICRO 2,048; Stream1 concurrency 4; Stream3 ring slots 4 and batch 196,608; shard count 4 and capacity 196,608 at scale 1,050,000 ppm; Stream4 batch/trigger 98,304 and four active sort slots.
- Static-hybrid history: 28 GiB RAM budget, 32 GiB disk budget, and 768 MiB GPU headroom.

## Raw run results

| Run | Wall time | Return codes | Peak process-tree RSS | RSS samples | Rank logs | Completion |
| --- | ---: | --- | ---: | ---: | --- | --- |
| first | 6.498522 s | torchrun 0; ranks 0/0 | 1,560,637,440 B | 556 total; 512 retained | stdout/stderr for ranks 0 and 1 | normal |
| collect A | 4.521576 s | torchrun 0; ranks 0/0 | 1,564,635,136 B | 235 total/retained | stdout/stderr for ranks 0 and 1 | `capacity_reached` |
| collect B | 4.463731 s | torchrun 0; ranks 0/0 | 1,564,192,768 B | 199 total/retained | stdout/stderr for ranks 0 and 1 | `capacity_reached` |

All before/after GPU snapshots contained exactly two Tesla T4 devices. The bounded combined logs were 719, 3,850, and 3,850 bytes. There was no overflow, OOM, timeout, collective hang, fatal runtime marker, traceback, or exception.

The real first-mode release record was:

```text
[default0]:puzzle_solved=1 puzzle_id=1 seconds=0.108281 solution_length=1 found_depth=1 touch_depth=0 solution=BR
```

The independent local replay against checked-in `data/test.csv` and `data/puzzle_info.json` confirms `BR` solves puzzle 1. Its token count is 1 and its depth decomposition is exactly `1 = 1 + 0`.

## Deterministic collection evidence

Both normal collect runs emitted the exact schema:

```text
puzzle_id depth_index found_depth total_depth known_length delta owner_rank solution_path
```

Each emitted 16 distinct, independently replayed valid paths. The two raw TSV files are byte-for-byte identical: 449 bytes, SHA-256 `74c12063c3f7cd6399546d6dd865d537e966bf8d9b935510174f9d46a92c748e`. Every row has puzzle id 1, token count equal to `total_depth`, known length 1, correct delta, a rank in `{0,1}`, and a depth index within the configured limit.

## Independent verification

- `python -m pytest tests/test_build_kaggle_cayleypy_task5_gate.py -q` -> `6 passed`.
- `python -m pytest -q` -> `134 passed`.
- Generated code-cell AST parsing, payload decompression, byte count, source SHA, private metadata, and package scan passed before push.
- `validate_gate_output(test_results/kaggle_cayleypy_task5_2xt4_gate_v2_2026-07-29)` -> `status=ok` with source, binary, and collection hashes above.
- A negative regression changes a remotely claimed CPU-valid path from `BR` to invalid `U`; the independent validator rejects it as CPU-invalid.
- A second explicit scan found no Windows private paths, feature-branch name, checkpoint source path, user name, credential prefix, API key, or authorization bearer field in the accepted evidence.
- Rank-0 failure injection remains `source-test-only-no-safe-runtime-hook`; no runtime hook or C++ algorithm was added.

## Evidence locations

- Accepted raw version-2 output: `test_results/kaggle_cayleypy_task5_2xt4_gate_v2_2026-07-29/`.
- Excluded version-1 diagnostic output: `test_results/kaggle_cayleypy_task5_2xt4_gate_v1_diagnostic_2026-07-29/`.
- Generated private package: `kaggle_cayleypy_task5_gate/`.
- Builder and independent validator: `tools/build_kaggle_cayleypy_task5_gate.py`.
- Local regressions: `tests/test_build_kaggle_cayleypy_task5_gate.py`.

No GitHub push, public publication, Task 6 work, Task 7 work, CUDA kernel change, Stream 1-5 algorithm change, or new runtime failure hook was performed.
