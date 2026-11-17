# CNOT Rust control collectives — remote checks

Source commit: `25fa9fc9e2372e97054f6f74b2b82ba7238c168b`

- CPU workflow [44199043726](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44199043726): passed Rust formatting, 3 `beam-core` CNOT tests, 4 `beam-runtime` CNOT tests, `cargo check -p beam-python`, 24 Python CNOT tests, and NCCL probe syntax compilation.
- CPU log artifact [14379159641](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44199043726/artifacts/14379159641), SHA-256 `c3b52e4450231d3fe19785717cefc66eb08e04d138465d716c8ed74e9ea0fc75`.
- Vast status workflow [44199043691](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44199043691): confirmed instance `61080396` was `exited`; the CUDA build and two-rank NCCL test were skipped. No start, stop, or rent request was made.
- Vast status artifact [14379992676](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44199043691/artifacts/14379992676), SHA-256 `c9038e1efb63a8aef3e0a924acdb639021fc0e73d919256d945fafaa958d1607`.

These checks validate CPU contracts and buildability only. CUDA compilation, NCCL collectives, and multi-GPU CNOT behavior remain unverified on hardware.
