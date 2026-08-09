# Stream 1 Branch Migration Verification

## Preserved source

- GitHub source branch: `codex/stream1-real-state-rank-parity`
- Preserved source tip: `f769e707dcda2b962bbaf99c62b40a1fd88f8108`
- Canonical local branch `stream1` was created directly at that object id.
- No source commit was rebased, squashed, rewritten, or cherry-picked.

The source history includes a merge of the public collector line. Keeping the
exact source tip was selected over replaying 176 mixed commits because exact
tree preservation is the no-regression migration gate. Future Stream 1 work
uses short task branches; historical subsystem separation is not manufactured
by rewriting already tested history.

## Python baseline

Command:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q tests/test_stream1_transformer_autotune.py tests/test_stream1_transformer_backends.py tests/test_stream1_transformer_exporter.py tests/test_stream1_transformer_parity.py tests/test_stream1_transformer_policy_from_cache.py tests/test_stream1_transformer_profile_summary.py tests/cayleypy_public
```

Result: `311 passed in 28.37s`, exit code `0`.

## CUDA/C++ baseline

- Host GPU: NVIDIA GeForce RTX 3070 Laptop GPU
- Host driver: `572.70`
- Container image: `gpu-dev-cutlass-nsight:2026-05-24`
- Container CUDA: `12.8.1`
- CUTLASS: `/opt/cutlass`
- Compile target: `sm_86`
- Fresh build directory: `build-branch-migration-sm86`

Commands:

```bash
cmake -S . -B build-branch-migration-sm86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBEAM_CUDA_ARCHITECTURES=86 \
  -DCUTLASS_DIR=/opt/cutlass
cmake --build build-branch-migration-sm86 -j2
ctest --test-dir build-branch-migration-sm86 --output-on-failure
```

Fresh verification result:

```text
ninja: no work to do.
100% tests passed, 0 tests failed out of 19
Total Test time (real) = 4.54 sec
```

This validates the local single-GPU sm86 tree. It does not replace live
multi-GPU T4/A100/H100, Kaggle, or SLURM acceptance; those remain separate
verification domains and their preserved evidence is unchanged.
