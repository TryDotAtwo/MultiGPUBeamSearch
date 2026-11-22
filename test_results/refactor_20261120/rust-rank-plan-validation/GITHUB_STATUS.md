# Rust rank-plan validation — hosted verification

Date: 2026-11-20  
Branch source: `a44be23b8587f5bb3affb9307bf0638a7162c534`

## Scope

The Python calibration validator dispatches ordinary typed rank-plan records
through optional PyO3 to the existing `beam_core::planning::validate_rank_plans`
contract. Old extensions and unsupported records retain the Python validator.
No search, quota, rank, collective, or CUDA behavior changes.

## Evidence

- RED: [run 44656721403](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44656721403) built the extension and ran the CPU suite: 261 passed, 2 skipped; the two expected failures showed that the PyO3 method and dispatch were missing.
- GREEN: [run 44658594054](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44658594054) passed Rust formatting/checks/tests, PyO3 build, 265 Python tests (2 skipped), and NCCL probe syntax.
- Actions artifact: [14594831023](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44658594054/artifacts/14594831023), SHA-256 `746312d931ec875bbceeb253b9ca5a316a47cbba788805109fda034eee8ccdd1`.

These are hosted CPU/PyO3 and syntax checks. No physical GPU or multi-GPU runtime was executed; no Vast instance was started.

- Main GREEN: [run 44659065473](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44659065473) on `main` commit `62feb392b66bed588d66e0f94b4049f82f523c3e` passed 265 Python tests (2 skipped), Rust checks, PyO3 build, and NCCL probe syntax.
- Main artifact: [14594144027](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44659065473/artifacts/14594144027), SHA-256 `61f754fbc8f51c56ad3d1b4407d630a0350539198b443fc9cfa0b3dc06382e34`.
