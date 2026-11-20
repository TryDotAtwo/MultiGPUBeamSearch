# Native CNOT verifier acceptance

## Source
- Branch: `codex/beam-cnot-runtime-acceptance`
- Tested code: `814ddaa385b6f68941449e3bf17546e3fdf63edb`
- Changes: `tools/verify_cnot.py` can require `engine=rust` and the Rust-owned TQP scorer; both CNOT CPU tests and the installed-wheel workflow exercise the Rust engine.

## RED
- [CNOT CPU run 44819995307](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44819995307): expected failure on the new test-only commit. The CLI rejected `--engine rust` and help did not expose `--require-rust-scorer`; 308 tests passed, 2 skipped.
- This demonstrates that the tests detect the missing CLI contract.

## GREEN
- [CNOT CPU run 44820597906](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44820597906): success. Rust checks and PyO3 build passed; 310 Python tests passed, 2 skipped, 104 warnings; NCCL probe syntax passed. Artifact [14672625098](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44820597906/artifacts/14672625098), SHA-256 `1e520abb300f25bb14e80384832dda91f79582e1530030ec5941d1625e40425b`.
- [Installed Python wheel run 44820639095](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44820639095): PEP 517 wheel built and installed; smoke ran outside the checkout with `PYTHONPATH` unset and loaded `_beam_runtime.abi3.so` from site-packages; `pip check` passed. A public CNOT verifier call selected `engine=rust`, used CPU/Gloo, solved the n=144 fixture with length 2, and passed bitwise replay. Artifact [14671043639](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44820639095/artifacts/14671043639), SHA-256 `f0335a4828b4d416234b5d32ba36cbf4b1b2517bbd48f2f65b85a27c45a9f4b1`.

## Not covered
- This is CPU/Gloo evidence. The private TQP checkpoint was not loaded, and physical CUDA/NCCL with one and two GPUs was not run.
- The current Vast workflow's latest relevant run [44812797614](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797614) skipped device work because instance 61080396 was stopped. No instance state change or paid compute occurred.
- `VAST_API_KEY` had already been confirmed by the successful read-only status workflow [44815930306](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44815930306); no secret overwrite was needed.
