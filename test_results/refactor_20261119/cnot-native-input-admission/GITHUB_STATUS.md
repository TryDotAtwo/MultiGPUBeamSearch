# CNOT typed-input admission

Implementation: commit `2a727739e3ea2e56530c1bae49d50c21bf97bf18` on `codex/cnot-admission-owner`.

- RED: Actions run [44420817709](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44420817709) failed on the new assertion that the Rust controller bypasses the Python input builder (163 passed, 2 skipped). Remote artifact: 14482655492.
- Formatting follow-up: Actions run [44421729086](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44421729086) stopped at `rustfmt --check`; the call layout was corrected. No compilation or tests ran in that attempt.
- GREEN: Actions run [44422140414](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44422140414) passed Rust formatting/checks, PyO3 build, the selected CNOT CPU compatibility suite (164 passed, 2 skipped), and the NCCL probe syntax check.
- Green artifact: 14480614488, SHA-256 `368f3dca736109bbebe750b2a4017cf322f7a545faeee795db16e6c912afba7f`.
- Published main revision `eb58643834af29cb9fb39b6d50a37f48e2a53748` passed Actions run [44422680720](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44422680720) with the same CPU checks (164 passed, 2 skipped). Main artifact: 14482048754, SHA-256 `c2f1a2384390934e3696748912363bc036b6469f41e75d46242f75c31bebf897`.
- This verifies CPU/control-flow compatibility only. Physical CUDA/NCCL and multi-GPU CNOT execution were not run.
