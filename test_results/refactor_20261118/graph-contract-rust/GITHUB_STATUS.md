# Rust GraphContract CPU validation

- Source commit: `fc356aab38eb75f0bd32173c765044ba5a5738e9`.
- GitHub Actions run [44335585382](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44335585382): success on a GitHub-hosted Linux CPU runner.
- Rust checks passed: rustfmt, 3 `beam-core` CNOT tests, 9 `beam-core` search tests, 4 `beam-runtime` CNOT tests, 4 `beam-native-sys` scorer tests, and the `beam-python` check/build.
- Python integration passed: 129 passed, 2 skipped; NCCL probe syntax check passed. The PyO3 extension was built and imported by the suite.
- Artifact [14441861160](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44335585382/artifacts/14441861160), `cnot-tqp-scorer-cpu-validation`, SHA-256 `b50a75854f46fc34363189ddc93ccd43baba81156f2cc0ca87dc1ba7f7786cac`.
- The first run, [44333600886](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44333600886), reached test collection and found that pinned `cayleypy==0.1.0` lacks the private `cayleypy.models.checkpoint` module. The test now computes the documented canonical hash independently and retains the existing Rust session identity check.
- Scope is CPU/control flow. No CUDA kernel execution, NCCL runtime, or physical multi-GPU search was tested.
