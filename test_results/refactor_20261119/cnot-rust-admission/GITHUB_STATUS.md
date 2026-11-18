# CNOT Rust-owned admission — verification record

- Branch: `codex/cnot-rust-admission`
- Green source revision: `462f394a62be2e959f071efc7e596a528762d620`
- Test-first run `44403506598`: expected red result for the new controller contract (161 passed, 2 skipped, 2 failed because Rust admission was not implemented yet).
- Build run `44404591194`: caught two Rust temporary-lifetime errors in the digest helper; both were fixed by retaining the Python byte owners.
- Green branch run `44405036760`: Rust formatting/checks, PyO3 build, CNOT CPU compatibility suite (163 passed, 2 skipped), and NCCL probe syntax passed.
- Hosted artifact: `14474283716`, SHA-256 `137d8273de03b5329ce907b4be891594625bda79c46021831787ebb824771679`.
- Scope: Rust resolves the existing engine selection and computes the compatible admission digest; Python input preparation and legacy-extension fallback remain. Search behavior, precision, and collective order are unchanged.
- Published main revision: `30498f85222b26b179b6caa8cebc9fda7d6d58d7`. Main CPU workflow `44406246496` completed successfully; artifact `14476060013`, SHA-256 `edf29e53984ba3a0d6838183c5d78934b9a882d2a9cbe78594e8b31a3c3fa382`.
- Existing-instance workflow `44406246457` completed its read-only Vast status check successfully, confirming the repository secret `VAST_API_KEY` is available. Instance `61080396` was stopped, so source transfer, build, and two-rank CUDA/NCCL execution were skipped; no instance state change was attempted. Artifact `14474543989`, SHA-256 `0e174f3ed20b2259d17ba2a43348ba8e50e1c39ed80d557870eb671caa45a9de`.
- Physical CUDA/NCCL CNOT multi-GPU execution remains unverified.
