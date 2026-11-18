# CNOT Rust-owned admission — verification record

- Branch: `codex/cnot-rust-admission`
- Green source revision: `462f394a62be2e959f071efc7e596a528762d620`
- Test-first run `44403506598`: expected red result for the new controller contract (161 passed, 2 skipped, 2 failed because Rust admission was not implemented yet).
- Build run `44404591194`: caught two Rust temporary-lifetime errors in the digest helper; both were fixed by retaining the Python byte owners.
- Green branch run `44405036760`: Rust formatting/checks, PyO3 build, CNOT CPU compatibility suite (163 passed, 2 skipped), and NCCL probe syntax passed.
- Hosted artifact: `14474283716`, SHA-256 `137d8273de03b5329ce907b4be891594625bda79c46021831787ebb824771679`.
- Scope: Rust resolves the existing engine selection and computes the compatible admission digest; Python input preparation and legacy-extension fallback remain. Search behavior, precision, and collective order are unchanged.
- Physical CUDA/NCCL CNOT multi-GPU execution was not run by this CPU workflow. Main-branch verification is pending.