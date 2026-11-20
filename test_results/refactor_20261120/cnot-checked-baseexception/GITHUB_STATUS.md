# CNOT checked-stage BaseException coordination — status

Date: 2026-11-20

## Change

Source commit: `3f79af666dbd58cd9912d5cd003be8a4fd323de4`.

Python's compatibility checker and the Rust checked ABI now treat every
multi-rank `BaseException` as a rank-local stage failure before reporting it
through the existing vote. Single-rank behavior is unchanged. Added tests cover
Python fallback and the compiled Rust/PyO3 checker with a simulated two-rank
`KeyboardInterrupt`.

## Hosted checks

- CNOT CPU Actions run [44708561397](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44708561397) completed successfully: Rust checks, PyO3 build, and 271 Python tests passed (2 skipped).
- Artifact [14615563947](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44708561397/artifacts/14615563947), SHA-256 `5294dd904cc054b5bcff621de6f291f84b77bff1d875aaf50c9e3b145a5f7e17`.
- Existing-Vast status run [44708561389](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44708561389) authenticated the configured secret; the instance was `exited`, so source checkout, transfer, build and GPU test were skipped. No state change was attempted.
- Sanitized status artifact [14616430543](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44708561389/artifacts/14616430543), SHA-256 `727c95f07cb1c931629e0cb748aff1e839570dc71993d9efad11e47d63996ac8`.

## Acceptance boundary

The compiled Rust/PyO3 checker was exercised on hosted CPU with a simulated
two-rank world. There was no physical CUDA/NCCL run, because the configured
Vast instance was not running. This does not validate the complete CNOT
multi-GPU recovery path.
