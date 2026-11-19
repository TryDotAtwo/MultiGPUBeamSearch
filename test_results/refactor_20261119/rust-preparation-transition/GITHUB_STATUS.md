# Rust prepared-session transition

## Source

- Implementation: [`02afe96`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/02afe96dbba604f1863b5aa1217e4b90a0b9e1f7).
- Test fixture type fix: [`e3e20d2`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/e3e20d2091aca4d411d98219dae5630f278781dc).
- Rust formatting correction: [`8f2955c`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/8f2955c8c56829602124a7f1fcfd6b9f88fa3a89).
- Hosted run [44481115716](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44481115716) tested `main` at `8f2955c8c56829602124a7f1fcfd6b9f88fa3a89`.

## Results

- Rust formatting checks passed.
- `beam-core --test cnot`: 5 passed; `beam-core --test search`: 9 passed.
- Runtime CNOT checks: 4 passed. Resident search-session checks: 2 passed.
- Preparation identity checks: 2 passed. Native TQP scorer checks: 4 passed.
- `cargo check -p beam-python` and PyO3 extension build passed.
- Python CNOT compatibility: 174 passed, 2 skipped.
- NCCL probe syntax check passed; no NCCL collective was executed.
- Actions artifact [14507489644](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44481115716/artifacts/14507489644), SHA-256 `41f880aaae7397ced8b1b3c519376e7468e1bf4abdc5608702ae2aa59e115580`, contains the hosted job logs and test output.

Two earlier attempts exposed test-fixture issues: run 44480179007 found the center fixture used `Vec<u8>` where the descriptor requires `Vec<u32>`; run 44480917052 then required rustfmt formatting. Both were corrected before the successful run above.

## Scope

Rust now retains a compatible rank group or releases it before a changed Python preparation callback. Python still builds the preparation descriptor and performs model preparation. Search semantics, rank ordering, CUDA stages, and public results are unchanged.

This is hosted CPU/PyO3 evidence. Physical CUDA execution, NCCL collectives, and multi-GPU acceptance were not run. No Vast instance was started for this validation.
