# Stream1 LibTorch Benchmark Passes Smoke 2026-07-05

## Scope

- Add `--passes` to `stream1_transformer_libtorch_benchmark`.
- Add `--passes` plumbing to `tools/stream1_transformer_backends.py` for the explicit `libtorch` backend.
- Update the Kaggle LibTorch-only benchmark package to run repeated passes and report mean-best rows.
- No production runner, MLP, or default no-LibTorch path changes.

## Verification

Python checks:

```text
python tests/test_stream1_transformer_backends.py
.....
Ran 5 tests in 0.001s
OK

python tests/test_stream1_transformer_parity.py
.......
Ran 7 tests in 0.000s
OK

PYTHONPATH=. python tests/test_stream1_transformer_exporter.py
......
Ran 6 tests in 0.224s
OK
```

Package syntax:

```text
python_ast_json_ok=1
```

Opt-in LibTorch Docker build:

```text
cmake -S . -B /tmp/build-libtorch-passes-smoke \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake \
  -DBEAM_ENABLE_LIBTORCH_STREAM1=ON \
  -DCUTLASS_DIR=/opt/cutlass \
  -DBEAM_CUDA_ARCHITECTURES=75
cmake --build /tmp/build-libtorch-passes-smoke --target stream1_transformer_libtorch_benchmark -j2
[100%] Built target stream1_transformer_libtorch_benchmark
```

Default no-LibTorch Docker check:

```text
cmake -S . -B /tmp/build-no-libtorch-passes-check \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_DIR=/opt/cutlass \
  -DBEAM_CUDA_ARCHITECTURES=75
cmake --build /tmp/build-no-libtorch-passes-check --target contract_tests -j2
/tmp/build-no-libtorch-passes-check/contract_tests
contract_tests=pass
```

## Notes

The repeated-pass feature is benchmark/control-plane only. It exists to avoid selecting a production candidate from a single noisy Kaggle row. The current production recommendation remains explicit LibTorch eager; CUDA Graph remains an explicit mode but not the default assumption.