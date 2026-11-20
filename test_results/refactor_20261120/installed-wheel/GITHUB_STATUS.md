# Installed Python wheel validation

**Result:** PASS on the temporary branch and on `main`.

Source commit: `e8a819fa6a46845bdffad95857007aa5b374ba04`  
Runner: GitHub-hosted Ubuntu, Python 3.11.18, stable Rust, CPU PyTorch 2.9.0.

The Actions job built the package through the PEP 517 backend, installed the resulting `multigpubeamsearch-0.1.0.dev1` wheel with its dependencies, passed `pip check`, and ran the smoke script from `$RUNNER_TEMP` with `PYTHONPATH` removed.

Verified:

- Imports resolve to installed `site-packages` for `multigpubeamsearch`, `cayleypy_native`, and `_beam_runtime`.
- Public `beam_search`, `NativeOptions`, `NativeModel`, `NativeEnsemble`, and `NativeSession` exports agree across both package names.
- Rust ABI is 1; the extension exposes `TQPScorer` and `InferenceCalibration`.
- The wheel contains an executable `beam-runner`, which reports `resident-search-v1`.

Runs:

- [Branch run 44724395099](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44724395099), artifact 14622269822, SHA-256 `e9580e05d047a8f73dbf585b22023110192bdebb31e6fba348082194b4523cef`.
- [Main run 44727232048](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44727232048), artifact 14623599422, SHA-256 `0fd444baec3ae693e891a5fe74fe49b6bbe7d4a7a2ed752f32fcbd740d1739a0`.

The check covers packaging, imports, and the worker's protocol-info command. It does not execute a search, initialize CUDA, load a CNOT checkpoint scorer, or exercise NCCL/physical GPUs. Those acceptance steps remain open.
