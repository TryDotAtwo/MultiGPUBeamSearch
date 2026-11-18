# Rust CNOT input preparation

Verified published main commit: `ddc855d734c0897de6af6efef48a6d711a0be998`.

- Test-first run [44396359433](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44396359433) failed on the two expected missing-preparer assertions (159 passed, 2 skipped).
- Branch run [44397470650](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44397470650) passed Rust formatting/checks, the PyO3 build, the CNOT CPU compatibility suite, and NCCL probe syntax.
- Hosted log artifact [14472066732](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44397470650/artifacts/14472066732), SHA-256 `6dd94319971341e455a170f266fe490ee89318f04c9363a53d31529777b6cac7`.
- Published main run [44398322200](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44398322200) passed the same checks at commit `ddc855d734c0897de6af6efef48a6d711a0be998`.
- Main artifact [14470868398](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44398322200/artifacts/14470868398), SHA-256 `6cc1a5ca4352fd567e190a47ebbbbdb45f67f467fb0c56746c3a69ef1d72424b`.

The Rust adapter now owns packed layout/control validation, requests an independent contiguous snapshot from PyTorch, serializes that snapshot, and constructs `beam_core::cnot::CnotMatrix`. The Python public API delegates when supported and keeps compatibility paths for old extensions and explicit Python validation. No search algorithm, precision policy, CUDA kernel, or collective changed.

Physical CUDA/NCCL and multi-GPU CNOT execution remain unverified by this CPU workflow.
