# Rust CNOT top-level orchestration

Verified code/test head: `3a194d68567fa403e6b3f6007d04b35489e87679`.

- Test-first Actions run [44390118399](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44390118399) confirmed both new controller tests failed because the Rust entry point was absent (157 passed, 2 skipped).
- Implementation build run [44391645320](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44391645320) passed Rust checks and PyO3 build; two tests exposed a missing `CnotResult` fixture reference.
- Fixture run [44392161874](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44392161874) passed Rust checks and PyO3 build; one multi-rank fixture lacked the existing `longs` and `sum` methods.
- Green run [44392740368](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44392740368) passed Rust formatting/checks, PyO3 build, 159 Python tests (2 skipped), and NCCL probe syntax.
- Hosted log artifact [14469202830](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44392740368/artifacts/14469202830), SHA-256 `b2fb5768a81122c46f8629dda235e4f323eb31937ab7469b665ef7daaf192323`.
- Published main run [44393853310](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44393853310) passed the same checks: 159 passed, 2 skipped. Artifact [14469468491](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44393853310/artifacts/14469468491), SHA-256 `d6b88e8a2fb27c25555f028d35235fbac0a2202f7f3eb00601c14939cc55076b`.

The Rust entry point owns admission error coordination, input consensus, transport choice, and native/compatibility stage dispatch. Existing Python scorer and model callbacks, the Python reference engine, and old-extension fallback remain. No search algorithm, precision policy, or GPU kernel changed. Physical CUDA/NCCL and multi-GPU CNOT execution were not tested.
