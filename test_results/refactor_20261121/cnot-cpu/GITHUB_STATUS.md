# CNOT CPU compatibility validation

- Source commit: `a78150bbb5a5f03083626256191dd601ebbdd9f0`
- Workflow: [run 44835851032](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44835851032)
- Result: success on GitHub-hosted `ubuntu-latest`. Rust formatting checks, Cargo CNOT/core/runtime tests, `beam-python` check, and the PyO3 build all completed successfully.
- Python compatibility suite: `310 passed, 2 skipped, 104 warnings`. It selected two-rank Gloo failure and cancellation cases for CNOT setup, selection, exchange preparation/payload, and SIGINT. The NCCL probe received a syntax-only `py_compile` check.
- Scope: CPU/Gloo only. No physical CUDA/NCCL run or trained TQP checkpoint scorer was exercised. Rust emitted unused/dead-code warnings; pytest also reported deprecation warnings.
- Artifact: [cnot-rust-domain-fallback-cpu-validation.zip](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44835851032/artifacts/14678336960), artifact ID `14678336960`, SHA-256 `2f335561f309773ecd1df29c3a993d15f343cfb3de78d81d803c24d89bbafe9a`, 6,477 bytes.
