# Rust inference selection compatibility

Date: 2026-11-19  
Commit: `4569ee14d3e5c1a9c7215d49764fabce6e127975`

## Change

- Added `beam-core::inference::select_inference_compatible`, preserving the Python selector's profile rejection rules, median/MAD calculations, seeded CPython-compatible bootstrap, baseline recovery, and deterministic tie behavior. The existing Rust selector is unchanged.
- Exposed the compatible selector through optional PyO3 `inference_select_compatible`. Python marshals ordinary `Measurement` records to Rust; unusual values, missing extensions, or unsupported extensions use the existing Python selector.
- `autotune.select_inference` now tries the Rust adapter and keeps the previous Python implementation as fallback.

## Hosted verification

GitHub Actions [run 44554484032](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44554484032) completed successfully on commit `4569ee14d3e5c1a9c7215d49764fabce6e127975`.

- Rust formatting and scoped Rust checks passed.
- PyO3 extension build passed.
- CPU compatibility suite: 217 passed, 2 skipped. This includes four selector parity cases plus public Rust dispatch and Python fallback cases.
- The NCCL probe syntax check passed; it did not execute a multi-GPU collective.
- Artifact [14545390297](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44554484032/artifacts/14545390297), SHA-256 `2a956ebd62dacdfaff8c19a29560ee9b98950c0c0d3d7f9ce6f6ce0b761a8451`.

This is a behavior-preserving migration of inference selection logic. No GPU search stages, scoring, precision, or multi-GPU communication changed. Physical CUDA/NCCL acceptance remains unverified.
