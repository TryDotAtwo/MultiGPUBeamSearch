# Rust permutation depth control verification

- Source commit on `main`: `da1b71df776bc4287dd63aa2a7c556ede8cc1abb`.
- [Permutation depth control CPU validation](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44376398409): passed. Rust contract tests, formatting, `beam-python` and `beam-cli` checks passed; the C++ native ABI fixture compiled with warnings as errors and passed its persistent-context and blocking-CLI callback checks.
- [CNOT CPU compatibility](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44375983736): passed on `97b32b655c758ae10ba88fba1fb3317a3ca0eba7`, which contains the same runtime implementation. Rust CNOT checks, PyO3 build, Python compatibility tests, and the two-rank NCCL probe syntax check passed.
- CPU evidence does not include a CUDA runner build or physical multi-GPU/NCCL execution.
- Full logs remain in GitHub Actions. CPU validation artifact `14461213438`, SHA-256 `bfaaaeb92dee719ab945a94706cc354676454219498a45bacb808f30567a95d4` (expires 2026-11-25).

## ABI-1 compatibility and ABI-2 solve-bucket callback

- Source commit on `main`: `697c1d1437d2e4f39cf3cc9c1115280c9e865810`.
- [Permutation depth control CPU validation](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44384857755): passed. Rust formatting, runtime and native-sys callback tests, Python/CLI checks, and the C++ ABI fixture passed. ABI-1 retains 24/12-byte descriptors and its original checkpoints; ABI-2 covers solve-bucket callbacks. Artifact `14465660931`, SHA-256 `5491a3bd3ad43b5b819dead0be6a47c12c2ae4473c846a3d5c1f426daba28e66`.
- [CNOT CPU compatibility](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44384857776): passed at the same source commit. Rust checks, PyO3 build, Python compatibility tests, and NCCL probe syntax passed. Artifact `14464319713`, SHA-256 `cbb7173343f43963441d54278d81fd64dfc6b158653288b34c570e14e5455b45`.
- These CPU runs do not verify the CUDA production runner or physical multi-GPU/NCCL execution. Full logs remain in their GitHub Actions artifacts.
