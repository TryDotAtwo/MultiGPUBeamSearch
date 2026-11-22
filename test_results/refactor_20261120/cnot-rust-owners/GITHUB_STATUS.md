# Typed CPU CNOT owner assignment in Rust

Date: 2026-11-20

## Change

The typed CPU CNOT compatibility path assigns owners through `beam_core::cnot::owner_ranks` and a PyO3 adapter. The typed CUDA path still calls the existing CUDA owner operator. Python-only and older-extension fallback behavior remains available. This changes stage ownership only; it does not change rank partition semantics, state equality, scoring, search order, precision, CUDA kernels, or collectives.

## Hosted verification

- RED run [44674560289](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44674560289): expected failure at the regression that forbids the typed CPU path from calling the Python tensor owner helper; 268 passed, 1 failed, 2 skipped. Artifact 14602160860, SHA-256 `76726ff8bb49e3d56cf7acbf19476c4b6759d6ad552df3b24e3efc9b9a203508`.
- Formatting-only failure [44698622193](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44698622193) occurred before compilation/tests; formatting was corrected.
- GREEN run [44699020897](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44699020897), commit `285d8492e4ce1eb5f752d138850ff96171d9d2c9`: Rust CNOT checks, PyO3 extension build, 269 CPU/PyO3 tests passed, 2 skipped, 104 warnings, and NCCL probe syntax check passed.
- GREEN artifact 14612107333: SHA-256 `f367eb4217389be60ddaff4e8812565a0a26ecde879962e3ba8bb0e159d92ceb`.

## Limits

No physical CUDA execution, NCCL collective, or multi-GPU runtime was run. This validates CPU compatibility and compile paths only. No paid compute was used.


## Main verification

- Main commit: `db9577fcc61a3b57fae1ba0e1e2bc5c579e9b001`.
- GREEN Actions run [44699705418](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44699705418): Rust CNOT checks, PyO3 build, 269 CPU/PyO3 tests passed, 2 skipped, 104 warnings, and NCCL probe syntax passed.
- Main artifact 14612416568: SHA-256 `51be4e713b50c58e70e9a7b95ec055ab2f039420715dfe275f4e5d9c8a8ddfa6`.
- Existing-Vast workflow [44699705287](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44699705287) reported the instance stopped and skipped remote build/execution. No GPU runtime ran.
