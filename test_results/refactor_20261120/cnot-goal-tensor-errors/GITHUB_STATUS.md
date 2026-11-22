# CNOT goal error tensor diagnostics — hosted verification

Date: 2026-11-20  
Branch source: `2bfc67bb615ffaf0fd8864de92b0ed819c2fa9c4`

## Scope

After the existing CNOT goal vote detects a rank-local error, Rust now gathers
UTF-8 message lengths and padded bytes through fixed tensor collectives. The
successful goal vote and winner selection are unchanged. This path uses the
existing `World.gather_fixed` transport; older checked-stage APIs keep their
object-gather compatibility behavior.

## Evidence

- RED: [run 44660740001](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44660740001) passed 265 tests (2 skipped); the new test failed because goal diagnostics used `all_gather_object`.
- GREEN: [run 44661053113](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44661053113) passed Rust formatting/checks/tests, PyO3 build, 266 Python tests (2 skipped), and NCCL probe syntax.
- Actions artifact: [14595053280](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44661053113/artifacts/14595053280), SHA-256 `8057258b53a903863f37a50d27a157a064bcd14ab0f8507daca087d8a3bdddd9`.

The test exercises the two-rank diagnostic contract using CPU tensors. No
physical GPU or multi-GPU goal-error run was executed.

- Main GREEN: [run 44661706426](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44661706426) on `main` commit `080f98fd1968ca8af5fc1a18629efb7ab707b5db` passed Rust checks, PyO3 build, 266 Python tests (2 skipped), and NCCL probe syntax.
- Main artifact: [14595702908](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44661706426/artifacts/14595702908), SHA-256 `5132784e06da62aa6ee3faeeaff9ca014ddc9ed2865fe439c0fa929b656f1f2c`.
