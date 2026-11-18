# Rust-owned CNOT scorer — GitHub CPU validation

Source commit: `f762ab013d9120879b40f4b6992bf89eca6f5b48`

- CPU workflow [44210146770](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44210146770): passed Rust formatting, 3 `beam-core` CNOT tests, 4 `beam-runtime` CNOT tests, 4 native-sys scorer tests, `cargo check` and `cargo build -p beam-python`, 28 Python CNOT compatibility tests, and NCCL probe syntax compilation.
- Full log artifact [14385263763](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44210146770/artifacts/14385263763), SHA-256 `ebd343ddcdf9f7ca0a62ecd14b994293b3b720d2fb7c41f47565a57a3258b212`.
- The suite built/imported the real PyO3 extension and checked public scorer dispatch, default arguments, and rejection of an integer in the bool option. It did not load the scorer shared library or run a CUDA kernel, NCCL collective, or multi-GPU search.

These results verify CPU contracts and Rust/PyO3 buildability only. Physical CNOT scoring and multi-GPU acceptance remain unverified.
