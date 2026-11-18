# Rust permutation depth control verification

- Source commit on `main`: `da1b71df776bc4287dd63aa2a7c556ede8cc1abb`.
- [Permutation depth control CPU validation](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44376398409): passed. Rust contract tests, formatting, `beam-python` and `beam-cli` checks passed; the C++ native ABI fixture compiled with warnings as errors and passed its persistent-context and blocking-CLI callback checks.
- [CNOT CPU compatibility](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44375983736): passed on `97b32b655c758ae10ba88fba1fb3317a3ca0eba7`, which contains the same runtime implementation. Rust CNOT checks, PyO3 build, Python compatibility tests, and the two-rank NCCL probe syntax check passed.
- CPU evidence does not include a CUDA runner build or physical multi-GPU/NCCL execution.
- Full logs remain in GitHub Actions. CPU validation artifact `14461213438`, SHA-256 `bfaaaeb92dee719ab945a94706cc354676454219498a45bacb808f30567a95d4` (expires 2026-11-25).
