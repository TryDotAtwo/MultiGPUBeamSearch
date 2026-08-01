# Stream1 Transformer Score-Key Digest Parity

Date: 2026-07-05
Branch: `codex/stream1-piece-transformer`
Base commit before this change: `6e38d59 Add Stream1 transformer parity gate`

## Scope

- Improve backend parity readiness without changing production solver dispatch.
- Add a full-output deterministic digest to the existing benchmark row contract.
- Preserve explicit backend selection: `pytorch`, `libtorch`, `native_cutlass`.
- No fallback behavior, no distillation, no MLP/default path changes.

## Digest Contract

All three benchmark paths now emit:

```text
score_key_digest=<uint64>
```

The digest is FNV-1a 64-bit over every quantized score key, in batch-major then move-major order. Each score key is consumed as a little-endian uint32 value. This catches row permutation/local drift that `checksum` plus `first_score_keys` can miss.

## Changed Files

- `tools/stream1_transformer_torch_benchmark.py`
- `tools/stream1_transformer_libtorch_benchmark.cpp`
- `tools/stream_benchmark_transformer.cu`
- `tools/stream1_transformer_parity.py`
- `tests/test_stream1_transformer_parity.py`

## Verification

```text
python -c "import ast, pathlib; files=['tools/stream1_transformer_torch_benchmark.py','tools/stream1_transformer_backends.py','tools/stream1_transformer_parity.py','tests/test_stream1_transformer_backends.py','tests/test_stream1_transformer_parity.py']; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in files]; print('python_ast_ok=1')"
python_ast_ok=1
```

```text
python tests\test_stream1_transformer_backends.py
.....
----------------------------------------------------------------------
Ran 5 tests in 0.000s

OK
```

```text
python tests\test_stream1_transformer_parity.py
....
----------------------------------------------------------------------
Ran 4 tests in 0.001s

OK
```

```text
$env:PYTHONPATH='.'
python tests\test_stream1_transformer_exporter.py
......
----------------------------------------------------------------------
Ran 6 tests in 0.225s

OK
```

```text
python tools\stream1_transformer_parity.py --weight-dir weights_fp16 --build-dir build-smoke --backends pytorch,libtorch,native_cutlass --batch 256 --out-dir test_results\stream1_transformer_parity_digest_dry_2026-07-05 --dry-run
stream1_transformer_parity_dry_run backend=pytorch mode=eager
stream1_transformer_parity_dry_run backend=libtorch mode=eager
stream1_transformer_parity_dry_run backend=native_cutlass mode=eager
stream1_transformer_parity_status=dry_run
stream1_transformer_parity_report=test_results\stream1_transformer_parity_digest_dry_2026-07-05\stream1_transformer_parity.md
```

```text
docker run --rm --gpus all -v "D:/100XH100/.worktrees/stream1-piece-transformer:/work" -w /work gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake -S . -B build-no-libtorch-verify -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75 >/tmp/cmake.log && cmake --build build-no-libtorch-verify --target contract_tests stream_benchmark -j2 >/tmp/build.log && ./build-no-libtorch-verify/contract_tests"
contract_tests=pass
```

```text
docker run --rm -v "D:/100XH100/.worktrees/stream1-piece-transformer:/work" -w /work cmz-native-dev:2026-05-26 bash -lc "cmake -S . -B build-libtorch-verify -DBEAM_ENABLE_LIBTORCH_STREAM1=ON -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75 >/tmp/cmake_libtorch.log && cmake --build build-libtorch-verify --target stream1_transformer_libtorch_benchmark -j2 >/tmp/build_libtorch.log && echo libtorch_target_build=pass"
libtorch_target_build=pass
```

Temporary build directories were removed after verification.