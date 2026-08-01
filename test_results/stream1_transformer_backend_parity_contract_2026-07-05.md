# Stream1 Transformer Backend Parity Contract

Date: 2026-07-05
Branch: `codex/stream1-piece-transformer`

## Scope

- Keep three explicit Stream1 `piece_transformer` backends: `pytorch`, `libtorch`, `native_cutlass`.
- Add a parity launcher that compares backend score-key output on a deterministic synthetic state batch.
- Add dry-run parity validation for clean checkouts without requiring CUDA binaries or exported weights.
- Preserve no-fallback/no-distillation policy and keep the MLP/default path untouched.

## Code Changes

- `tools/stream1_transformer_parity.py`: new parity/dry-run tool.
- `tools/stream1_transformer_backends.py`: added PyTorch reference control and native synthetic-state env support.
- `tools/stream1_transformer_torch_benchmark.py`: emits `checksum` and `first_score_keys`; can skip fixture reference validation for synthetic parity runs.
- `tools/stream1_transformer_libtorch_benchmark.cpp`: emits `first_score_keys` alongside checksum.
- `tools/stream_benchmark_common.hpp` / `tools/stream_benchmark.cu`: deterministic synthetic state batch support via `BEAM_STREAM1_SYNTHETIC_STATES=1`.
- `tools/stream_benchmark_transformer.cu`: native transformer microbenchmark emits checksum and first score keys in stdout/report.

## Verification

```text
python -c "import ast, pathlib; files=['tools/stream1_transformer_torch_benchmark.py','tools/stream1_transformer_backends.py','tools/stream1_transformer_parity.py','tests/test_stream1_transformer_backends.py']; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in files]; print('python_ast_ok=1')"
python_ast_ok=1
```

```text
python tests\test_stream1_transformer_backends.py
.....
----------------------------------------------------------------------
Ran 5 tests in 0.001s

OK
```

```text
$env:PYTHONPATH='.'
python tests\test_stream1_transformer_exporter.py
......
----------------------------------------------------------------------
Ran 6 tests in 0.253s

OK
```

```text
python tools\stream1_transformer_backends.py --backend native_cutlass --mode eager --build-dir build-smoke --weight-dir weights_fp16 --b-micro 256 --concurrency 1 --synthetic-states --dry-run --json
```

Result: command/env JSON includes `BEAM_STREAM1_SYNTHETIC_STATES=1`, `BEAM_STREAM1_TRANSFORMER_B_MICRO=256`, and the expected native transformer benchmark env fields.

```text
python tools\stream1_transformer_backends.py --backend pytorch --weight-dir weights_fp16 --dry-run --json
```

Result: command JSON includes `--skip-reference` for synthetic benchmark/parity usage.

```text
python tools\stream1_transformer_parity.py --weight-dir weights_fp16 --build-dir build-smoke --backends pytorch,libtorch,native_cutlass --batch 256 --out-dir test_results\stream1_transformer_parity_dry_2026-07-05 --dry-run
stream1_transformer_parity_dry_run backend=pytorch mode=eager
stream1_transformer_parity_dry_run backend=libtorch mode=eager
stream1_transformer_parity_dry_run backend=native_cutlass mode=eager
stream1_transformer_parity_status=dry_run
stream1_transformer_parity_report=test_results\stream1_transformer_parity_dry_2026-07-05\stream1_transformer_parity.md
```

## Notes

- No GPU parity execution was run in this verification pass because the local clean worktree does not keep the Kaggle transformer weights or built backend binaries permanently.
- The parity tool is ready for GPU hosts: after exported weights and C++ binaries exist, run the same command without `--dry-run`.