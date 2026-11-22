# Rust NativeOptions validation — hosted CPU evidence

- Code under test: `dd81bec988a7e2fc66996200f304cc0b3371604c`.
- Workflow: [CNOT CPU compatibility run 44225114516](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44225114516) — success.
- Rust checks: core CNOT/search, runtime CNOT, and native-sys TQP scorer tests; `cargo check -p beam-python`; `cargo build -p beam-python`.
- Python integration: 121 passed, including option validation, old/missing/unloadable extension fallback, Rust control adapters, CNOT CPU paths, and public index-sort configuration.
- NCCL: two-rank probe syntax check passed; no collective was executed.
- Artifact: [14391754257](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44225114516/artifacts/14391754257), SHA-256 `44e646ba4a7c5d264baad8c54bd86fce06988ea5b2a97ea71f3fe308160ef0aa`.
- Not verified: CUDA execution, physical multi-GPU behavior, Compute Sanitizer, performance, Kaggle runtime.
- Diagnostic red runs: 44223487738 found falsey `None` being omitted from the Rust descriptor; 44224826858 demonstrated the missing import-error fallback. Both are covered by the final passing run.
