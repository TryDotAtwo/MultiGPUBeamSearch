# Rust finalizes resident search outcomes

## Source and expected RED

- Metadata-cache implementation: [0581fb5](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/0581fb5a83c9885f452f50316c6476c32be1977a).
- Test-only finalization checks: [616f026](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/616f026eac013826deb85f94e52acc886093f82b) and formatting correction [5861e1f](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/5861e1f6a896be2cb2ce4984f1d9cf324c740c77).
- The Python RED run [44490715135](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44490715135) found the callback metadata lacked persistent_session (2 failed, 77 passed, 1 skipped).
- After formatting, Rust runtime RED [44491198816](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44491198816) compiled and failed on the expected missing callback marker in native-outcome.json. The earlier run [44490715141](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44490715141) stopped first on rustfmt.

## Implementation and GREEN

- Rust finalization implementation: [9a9064e](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/9a9064e9984d9ba304b0f37533cbc8e63610f865).
- [Rust public owner, run 44492514040](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44492514040): 79 passed, 1 skipped. Artifact [14513721658](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44492514040/artifacts/14513721658), SHA-256 b7e4900cff5818266eec3d83ca884f65600ab02cc77bf734f0c348dea8df260e.
- [Permutation depth control, run 44492513834](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44492513834): Rust test groups 5 and 2 passed; rustfmt, cargo checks and CPU native ABI fixture passed. Artifact [14514590009](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44492513834/artifacts/14514590009), SHA-256 eb29091d3a5dbcf55a5bcb981af2923d8f36dd461b9b01466cafe036f6ebf2b8.
- [CNOT CPU compatibility, run 44492513959](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44492513959): Rust formatting; beam-core CNOT/search 5 and 9 passed; runtime CNOT 4 and search-session 2 passed; PyO3 build/check passed; Python compatibility 174 passed, 2 skipped; NCCL probe syntax passed. Artifact [14513123280](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44492513959/artifacts/14513123280), SHA-256 6f8a7c89e53cf30b1f73571a5b770575500fd1abe12b7f367d3c19bcf1a9a3e3.

## Scope and evidence limits

Rust retains the complete callback metadata and atomically finalizes the most recent native-outcome.json after the callback supplies it. Repeated Rust-owned outcomes are consumed once and stay bound to their run directory. NativeSession builds persistent_session metadata before forwarding the result to Rust; on successful finalization it does not repeat the metadata merge or rewrite the outcome from Python. Older or unable-to-finalize PyO3 extensions keep the Python cache/write fallback.

Search order, inference kernels, rank exchange, CNOT semantics and public result formats are unchanged. This is hosted CPU/PyO3 evidence only: no physical CUDA/NCCL collective or multi-GPU acceptance run occurred, and no Vast instance was started.
