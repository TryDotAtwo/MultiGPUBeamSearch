# Rust-owned CNOT scorer — GitHub CPU validation

Source commit: `c062df51d6c35df0248018efae83ead0b5a12c86`

- CPU workflow [44208783672](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44208783672): passed Rust formatting, 3 `beam-core` CNOT tests, 4 `beam-runtime` CNOT tests, 4 native-sys scorer tests, `cargo check -p beam-python`, 27 Python CNOT compatibility tests, and NCCL probe syntax compilation.
- Full log artifact [14384877235](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44208783672/artifacts/14384877235), SHA-256 `83510320d4f985701d3ab2bb43243dce2aa1d119574eb6bc292d5b17253cb5e7`.
- The Python dispatch test selects a mocked Rust constructor. The scorer shared library was not loaded, and no CUDA kernel, NCCL collective, or multi-GPU search ran.

These results verify CPU contracts and Rust/PyO3 buildability only. Physical CNOT scoring and multi-GPU acceptance remain unverified.
