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

## Main verification after merge
- Squash commit: `ecfc4d4ad6084a72ecb34a9a958c3161b143d9c8`.
- [CNOT CPU run 44821882474](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44821882474): success, 310 passed, 2 skipped, 104 warnings. Artifact [14672636499](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44821882474/artifacts/14672636499), SHA-256 `37697d7e3dac3e097e79a89c1d4d6a3ee685ac5cf6b5d663bb0ee8d452decc50`.
- [Installed wheel run 44821882308](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44821882308): success. PEP 517 wheel build/install, outside-checkout smoke, `pip check`, and the Rust CNOT search all passed. Artifact [14672282878](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44821882308/artifacts/14672282878), SHA-256 `816f1d22a449a47d8b995374eafc4c25b639e5a01fc6fb1427cc0c4130214b9a`.

## Not covered
- This is CPU/Gloo evidence. The private TQP checkpoint was not loaded, and physical CUDA/NCCL with one and two GPUs was not run.
- The current Vast workflow's latest relevant run [44812797614](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797614) skipped device work because instance 61080396 was stopped. No instance state change or paid compute occurred.
- `VAST_API_KEY` had already been confirmed by the successful read-only status workflow [44815930306](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44815930306); no secret overwrite was needed.
