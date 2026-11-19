# Rust-owned CNOT identity and replay

- Verified commit: `f2cf3ec0d9cd8eb66381459952b4ce93ff5192da`
- GitHub Actions run: [44426467664](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44426467664) — success
- Coverage: Rust formatting/checks, 5 packed CNOT core tests, Rust CNOT controller tests, PyO3 build, 165 Python compatibility tests (2 skipped), NCCL probe syntax.
- Artifact: [cnot-owned-matrix-cpu-validation](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44426467664/artifacts/14485300094), ID `14485300094`, SHA-256 `e11a6d4ccf9ee0203fee83c8a584606c7290ef1e056e5e4f7ed7e1632f9f1c20`.
- Test-first run [44425293831](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44425293831) confirmed the pre-change call to `torch.equal`; it reported 164 passed and 2 skipped plus the intended new failure.
- Physical CUDA/NCCL and multi-GPU CNOT execution were not run. No build outputs or full logs were stored locally.
