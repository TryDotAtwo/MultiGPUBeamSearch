# Rust-owned CNOT scorer — GitHub CPU validation

Latest CI source commit: `2dc542cc2621164db8e9166219e59168ee1c0363`

- CPU workflow [44210146770](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44210146770): passed Rust formatting, 3 `beam-core` CNOT tests, 4 `beam-runtime` CNOT tests, 4 native-sys scorer tests, `cargo check` and `cargo build -p beam-python`, 28 Python CNOT compatibility tests, and NCCL probe syntax compilation.
- Full log artifact [14385263763](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44210146770/artifacts/14385263763), SHA-256 `ebd343ddcdf9f7ca0a62ecd14b994293b3b720d2fb7c41f47565a57a3258b212`.
- The suite built/imported the real PyO3 extension and checked public scorer dispatch, default arguments, and rejection of an integer in the bool option. It did not load the scorer shared library or run a CUDA kernel, NCCL collective, or multi-GPU search.

## Rust-native stage scorer dispatch

- CPU workflow [44212845662](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44212845662) passed Rust formatting, 3 `beam-core` CNOT tests, 4 `beam-runtime` CNOT tests, 4 native-sys scorer tests, `cargo check`, `cargo build -p beam-python`, 33 Python tests, and NCCL probe syntax compilation.
- Added two focused tests that invoke `run_cnot_native` on CPU with a Python `WireFixture`, covering the Python scorer fallback through Rust-owned stages and batch operations.
- Full log artifact [14387022871](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44213429756/artifacts/14387022871), SHA-256 `c2c1b383d37a36357db0ed4f057661a24fb04644646538dd061249945a6e26d7`.
- The built-in TQP scorer direct Rust method path is compile-checked but was not invoked: the workflow has CPU PyTorch and no checkpoint scorer library. CUDA scoring and physical multi-GPU acceptance remain pending.
- Vast status workflow [44212845584](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44212845584) confirmed the owned instance was stopped. Its build and two-rank NCCL steps were skipped without changing instance state; status artifact [14387096650](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44212845584/artifacts/14387096650), SHA-256 `bb9e37116717c2cf62d220727236f9872782d68d89d00112791f411446243895`.

These results verify CPU contracts and Rust/PyO3 buildability only. Physical CNOT scoring and multi-GPU acceptance remain unverified.
