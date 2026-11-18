# CNOT Rust control collectives — remote checks

Source commit: `25fa9fc9e2372e97054f6f74b2b82ba7238c168b`

- CPU workflow [44199043726](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44199043726): passed Rust formatting, 3 `beam-core` CNOT tests, 4 `beam-runtime` CNOT tests, `cargo check -p beam-python`, 24 Python CNOT tests, and NCCL probe syntax compilation.
- CPU log artifact [14379159641](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44199043726/artifacts/14379159641), SHA-256 `c3b52e4450231d3fe19785717cefc66eb08e04d138465d716c8ed74e9ea0fc75`.
- Vast status workflow [44199043691](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44199043691): confirmed instance `61080396` was `exited`; the CUDA build and two-rank NCCL test were skipped. No start, stop, or rent request was made.
- Vast status artifact [14379992676](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44199043691/artifacts/14379992676), SHA-256 `c9038e1efb63a8aef3e0a924acdb639021fc0e73d919256d945fafaa958d1607`.

These checks validate CPU contracts and buildability only. CUDA compilation, NCCL collectives, and multi-GPU CNOT behavior remain unverified on hardware.

## Rust control-collective dispatch — 2026-11-18

Source commit: `4be8bb79f0681430c7d1772dd944a716e6041ad0`

- Test-first CPU workflow [44356451066](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44356451066): expected failure in the new Rust ABI and dispatch checks (2 failed, 138 passed, 2 skipped).
- Green CPU workflow [44357542114](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44357542114): Rust CNOT checks, PyO3 check/build, 140 Python tests passed, 2 skipped, and NCCL probe syntax check passed.
- Full hosted test log artifact [14452657387](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44357542114/artifacts/14452657387), SHA-256 `6390bd7da9d7cf023985740c579ca03497d1c865e74d52b4bbae29c6de636d1d`.
- Vast status workflow [44357542194](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44357542194): confirmed the existing instance was stopped and skipped transfer/build/NCCL execution without changing its state.

The implementation moves control dispatch into Rust while reusing the existing Python/PyTorch transport. No CUDA/NCCL runtime or physical multi-GPU search was executed.
