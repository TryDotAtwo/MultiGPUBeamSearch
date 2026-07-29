# Task 5 private Kaggle real-2xT4 acceptance gate

Date: 2026-07-29

## Outcome

Private Kaggle kernel `trydotatwo/cayleypy-public-task-5-2xt4-gate`, version 3, completed successfully and passed the independent downloaded-evidence plus remote-attestation validator. The accepted run used exactly two Tesla T4 GPUs, the tracked output-dimension-24 fp16 weights, beam 65,536, depth limit 8, puzzle 1, and zero K1/K2 radii. It ran first mode once and collect mode twice with fresh run/history directories.

Version 1 was diagnostic-only and failed the exact-slug provenance gate. Version 2 was the first green residual acceptance. Version 3 supersedes it after closing two independent-review P2s in notebook/gate-only scope: EOF-tail output now uses the same byte cap as live capture, every capture/read failure performs bounded process-group termination and reap with kill fallback, and acceptance now requires external Kaggle push/status/list/pull evidence instead of notebook self-attestation.

## Review-P2 closure and remote attestation

- Capture TDD RED: two focused failures (`DID NOT RAISE` for an oversized EOF tail; no TERM/KILL calls after a read exception). GREEN: both adversarial generated-cell tests pass and assert bounded waits, selector closure, kill fallback, and reap.
- Remote-attestation TDD RED: the validator accepted drifted external evidence. GREEN: one adversarial test rejects wrong slug, pushed v2, public metadata, non-COMPLETE status, invalid `lastRunTime` ordering, and a semantically different pulled notebook.
- Full raw remote evidence is SHA-256 bound in `remote/capture_manifest.json`: push receipt, status output, list CSV, pulled metadata, and pulled notebook. Known Kaggle CLI warning/blank lines are tolerated only around exactly one success/status/CSV record.
- Exact remote facts: pushed version 3; private `true`; status `COMPLETE`; remote `lastRunTime=2026-07-29T01:28:46.683000Z`; completion observed `2026-07-29T01:31:29.760778Z`.
- Kaggle rewrote notebook cell `source` arrays as strings on pull. Raw hashes differ, but after only that representation normalization the pulled notebook is JSON-identical to the pushed package.

## Package and provenance

- Private kernel version: 3 (`COMPLETE`; remote last run UTC `2026-07-29T01:28:46.683000Z`).
- Generated/pushed notebook: 85,360 bytes, SHA-256 `f817a7be1a848918b685e9cb23a0b6c3d5508eeb498c178a1fa7ca08c151fd7a`.
- Kernel metadata SHA-256: `b9a22006ac26a28127cf0f1f1c162acf41350369ac0159bc0ffc5def97b9bc32`.
- Pulled notebook: 80,030 bytes, SHA-256 `4f7562f59ecc66355a9c885c3095de1921735072309a64a047298433d0175753`; semantic equality after source-array normalization passed.
- Pulled metadata SHA-256: `ca9203096b537f8969c7cfa1d1c8cbbc13f1e0a639b17cf9f1dd6af05536184c`.
- Raw push receipt SHA-256: `350210f15d60f73c345bce5736b0eeaf65876646e5118db241b2fd655db1af1f`.
- Raw COMPLETE status SHA-256: `7f8321806701934d01dec5f3deaaa76fba5709b882ebb4d9ff3c982254f3195a`.
- Raw list CSV SHA-256: `bbfa98519960931958b88d86a2f757be956b68ba29021dd6cef14d6fe0966295`.
- Public base `origin/main`: `6f95bd6bdb32b5f6ef7cca32b96967bce6036503`.
- Reviewed source commit: `6830401ed2086921d2563c2bc3c11faf6c5a0741`.
- Embedded/overlaid `production_runner.cu`: 242,054 bytes, SHA-256 `f7d20a2fdec5748052b09804a2b2878cb13f854b8dd29e05db92d6828c223774`.
- CUTLASS checkout: `afa1772203677c5118fcd82537a9c8fefbcc7008`.
- Release SM75 binary: 6,100,856 bytes, SHA-256 `8dd39f5539fbf88f1d47128834f16e3d66760ef7ce74047758d4d6579b57d53e`.
- Build attestation: `Release`, CUDA architecture `75`, NCCL dynamically linked.
- Build times: base clone 4.433524 s, CUTLASS clone 2.745559 s, configure 4.440432 s, compile 67.783181 s.
- Raw build log SHA-256: `822d7688f302b6042654123c93d3053fd03fec4fa821c5db43f86b8cd6ea442e`.
- Raw source manifest SHA-256: `36eda978eba823880abd75a39b4e90775f6b6710fac6d4c9a8bf70358c6d20f7`.
- Raw gate summary SHA-256: `37554994172f765319502eb9bff318d525cfa7ad4b5aae7d7260a6d601a2c551`.

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
| first | 6.529306 s | torchrun 0; ranks 0/0 | 1,565,368,320 B | 491 total/retained | stdout/stderr for ranks 0 and 1 | normal |
| collect A | 4.720228 s | torchrun 0; ranks 0/0 | 1,564,393,472 B | 203 total/retained | stdout/stderr for ranks 0 and 1 | `capacity_reached` |
| collect B | 4.741107 s | torchrun 0; ranks 0/0 | 1,567,088,640 B | 174 total/retained | stdout/stderr for ranks 0 and 1 | `capacity_reached` |

All before/after GPU snapshots contained exactly two Tesla T4 devices. The bounded combined logs were 720, 3,850, and 3,850 bytes. There was no overflow, OOM, timeout, collective hang, fatal runtime marker, traceback, or exception.

The real first-mode release record was:

```text
[default0]:puzzle_solved=1 puzzle_id=1 seconds=0.095335 solution_length=1 found_depth=1 touch_depth=0 solution=BR
```

The independent local replay against checked-in `data/test.csv` and `data/puzzle_info.json` confirms `BR` solves puzzle 1. Its token count is 1 and its depth decomposition is exactly `1 = 1 + 0`.

## Deterministic collection evidence

Both normal collect runs emitted the exact schema:

```text
puzzle_id depth_index found_depth total_depth known_length delta owner_rank solution_path
```

Each emitted 16 distinct, independently replayed valid paths. The two raw TSV files are byte-for-byte identical: 449 bytes, SHA-256 `74c12063c3f7cd6399546d6dd865d537e966bf8d9b935510174f9d46a92c748e`. Every row has puzzle id 1, token count equal to `total_depth`, known length 1, correct delta, a rank in `{0,1}`, and a depth index within the configured limit.

## Independent verification

- `python -m pytest tests/test_build_kaggle_cayleypy_task5_gate.py -q` -> `9 passed`.
- `python -m pytest -q` -> `137 passed`.
- Generated code-cell AST parsing, payload decompression, byte count, source SHA, private metadata, and package scan passed before push.
- `validate_gate_output(test_results/kaggle_cayleypy_task5_2xt4_gate_v3_2026-07-29)` -> `status=ok` with source, binary, collection, external remote, and pulled-notebook attestations above.
- A negative regression changes a remotely claimed CPU-valid path from `BR` to invalid `U`; the independent validator rejects it as CPU-invalid.
- A second explicit scan found no Windows private paths, feature-branch name, checkpoint source path, credential prefix, API key, or authorization bearer field in the accepted evidence; only the expected Kaggle ref/title/author metadata remains.
- Rank-0 failure injection remains `source-test-only-no-safe-runtime-hook`; no runtime hook or C++ algorithm was added.

## Evidence locations

- Accepted raw version-3 output and sanitized remote evidence: `test_results/kaggle_cayleypy_task5_2xt4_gate_v3_2026-07-29/`.
- Superseded green version-2 output: `test_results/kaggle_cayleypy_task5_2xt4_gate_v2_2026-07-29/`.
- Excluded version-1 diagnostic output: `test_results/kaggle_cayleypy_task5_2xt4_gate_v1_diagnostic_2026-07-29/`.
- Generated private package: `kaggle_cayleypy_task5_gate/`.
- Builder and independent validator: `tools/build_kaggle_cayleypy_task5_gate.py`.
- Local regressions: `tests/test_build_kaggle_cayleypy_task5_gate.py`.

No GitHub push, public publication, Task 6 work, Task 7 work, CUDA kernel change, Stream 1-5 algorithm change, or new runtime failure hook was performed.
