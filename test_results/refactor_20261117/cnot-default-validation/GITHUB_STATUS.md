# CNOT default Rust validation: CPU verification

Status: source-pinned CPU checks passed. No CUDA GPU test or Vast instance was run for this change.

## RED then GREEN

- RED: [Actions run 44150899645](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44150899645) failed only the two new cases for the missing `auto` behavior; 17 existing Python tests passed. The earlier runner setup failure due to missing NumPy was fixed before this RED run.
- GREEN: [Actions run 44153448093](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44153448093) passed.

## GREEN checks

- `cargo test --locked -p beam-core --test cnot`: 3 passed.
- `cargo test --locked -p beam-runtime cnot_search`: 3 relevant tests passed.
- `cargo check --locked -p beam-python`: passed, including the PyO3 binding.
- `PYTHONPATH=integrations/cayleypy_native python -m pytest -q integrations/cayleypy_native/tests/test_cnot_search.py`: 19 passed, including the two-rank CPU Gloo case.
- GitHub Actions artifact: [cnot-cpu-validation, artifact 14360561194](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44153448093/artifacts/14360561194), SHA-256 `b80e09b1541413b011cdd47f95e6388a4b0d086d4d27de57e9269fdc6c83df83`.

This confirms Rust core and PyO3 compilation plus CPU semantics. It does not establish CUDA execution, NCCL behavior, or multi-GPU hardware parity for the new default path.
