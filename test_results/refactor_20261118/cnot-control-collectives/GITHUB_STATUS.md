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

## Rust row-exchange dispatch — 2026-11-18

Source commit: `6d410739ab5c8ce8deb0f09d1f42cb8e1020cee7`

- Test-first CPU workflow [44359803108](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44359803108): expected failure in the new ABI and `World.exchange` dispatch checks (2 failed, 143 passed, 2 skipped).
- Green CPU workflow [44360850283](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44360850283): Rust formatting/checks, PyO3 build, 145 Python tests passed, 2 skipped, and NCCL probe syntax passed. The included two-rank Gloo test checks row order, empty splits, and process-group reuse.
- Full hosted test log artifact [14456055164](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44360850283/artifacts/14456055164), SHA-256 `308e8d044b951c8f699acee98b54f07a97fea756da29d8cb100c66ef7c6ab35e`.
- Vast status workflow [44360849887](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44360849887): instance `61080396` was exited; transfer/build/NCCL were skipped without starting, stopping, or renting compute.

The Rust layer now owns this exchange dispatch but still calls the existing PyTorch transport or CUDA operator. Physical CUDA/NCCL and multi-GPU CNOT runtime remain unverified.

## Rust transport selection — 2026-11-18

Source commit: `ee162cc9b8c5c612d62ba8ac432a647d175cfc91`

- Test-first CPU workflow [44362409466](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44362409466): expected failures in the new Rust ABI and selection assertions (3 failed, 146 passed, 2 skipped).
- Green CPU workflow [44363394178](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44363394178): Rust formatting/checks, PyO3 check/build, 149 Python tests passed, 2 skipped, and NCCL probe syntax passed.
- Full hosted test log artifact [14455567451](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44363394178/artifacts/14455567451), SHA-256 `0f42f0b4e21e7e292538f4681f56b6d3e0bcbb1e8a5728fcd3f9108c56c88724`.
- Vast status workflow [44363394309](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44363394309): instance `61080396` was exited; transfer/build/NCCL were skipped without changing instance state.

The Rust layer now owns the transport choice and rank-consensus control flow. Existing Rust and Python transport implementations remain in use. This is CPU/control-flow evidence only; physical CUDA/NCCL and multi-GPU CNOT runtime remain unverified.
