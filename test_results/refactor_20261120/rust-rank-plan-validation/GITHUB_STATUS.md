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
