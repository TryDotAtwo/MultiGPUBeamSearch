# Rust-owned CNOT identity and replay

- Verified commit: `f2cf3ec0d9cd8eb66381459952b4ce93ff5192da`
- GitHub Actions run: [44426467664](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44426467664) — success
- Coverage: Rust formatting/checks, 5 packed CNOT core tests, Rust CNOT controller tests, PyO3 build, 165 Python compatibility tests (2 skipped), NCCL probe syntax.
- Artifact: [cnot-owned-matrix-cpu-validation](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44426467664/artifacts/14485300094), ID `14485300094`, SHA-256 `e11a6d4ccf9ee0203fee83c8a584606c7290ef1e056e5e4f7ed7e1632f9f1c20`.
- Test-first run [44425293831](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44425293831) confirmed the pre-change call to `torch.equal`; it reported 164 passed and 2 skipped plus the intended new failure.
- Published `main` at `b8615ba19eb6b7fa7e13dd89c7a1d0c4e22fbec8` passed run [44427129610](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44427129610): 165 passed, 2 skipped. Main artifact ID `14484482646`, SHA-256 `f936f740628e88e7d31006debab4a8b000e3038bd81ad73d053f69d6880af1f7`.
- Existing-Vast check [44427129602](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44427129602) reported instance `61080396` as `exited`; GPU/NCCL work was skipped without an instance state change.
- Physical CUDA/NCCL and multi-GPU CNOT execution were not run. No build outputs or full logs were stored locally.
