# Rust component pipeline-cache receipt validation

- Tested code commit: `d36d6b38e77c57815e8d93f2170462cf9b55eb22`.
- RED: [GitHub Actions run 44648572656](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44648572656) passed Rust checks and the PyO3 build, then failed at the new test because the Rust API was absent (248 passed, 2 skipped, 1 failed).
- GREEN: [GitHub Actions run 44649100197](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44649100197) passed Rust checks, PyO3 build, 249 Python tests (2 skipped), and NCCL probe syntax.
- Final branch check: [GitHub Actions run 44649561313](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44649561313) passed Rust checks, PyO3 build, 250 Python tests (2 skipped), and NCCL probe syntax. Artifact [14590465891](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44649561313/artifacts/14590465891), SHA-256 `9aff948e3aec5b88173fcb2121cae0789680e1625f037afb1c6185093f24de6b`.
- The component-calibrated receipt path now uses the optional Rust validator. Full-frontier receipts stay on the Python selector; older extensions keep the Python fallback. Fresh all-rank memory admission and geometry checks remain in place.
- These hosted CPU/PyO3 checks and NCCL probe syntax do not establish physical CUDA/NCCL multi-GPU runtime acceptance.
