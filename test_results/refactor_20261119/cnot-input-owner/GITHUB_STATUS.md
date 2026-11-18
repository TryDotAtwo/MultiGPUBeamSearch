# Rust-owned CNOT search input — verification record

- Test-first run `44407881591` failed for the expected missing `CnotSearchInput` type: 1 failed, 162 passed, 2 skipped.
- Green branch run `44408663743` passed Rust checks, PyO3 build, CNOT CPU compatibility tests (163 passed, 2 skipped), and NCCL probe syntax.
- Hosted artifact `14476796556`, SHA-256 `2f41fd59fbca781ad91269f8731f06544af62c0a7ba99bea15329eab5424d8ec`.
- Matching extensions now return a frozen `CnotSearchInput` carrying the validated `beam_core::cnot::CnotMatrix`, the same detached tensor snapshot, and raw bytes. The Python adapter passes this object through; four-item results from older extensions remain supported.
- Search, scorer, and collective behavior did not change. This CPU workflow does not verify physical CUDA/NCCL or multi-GPU CNOT execution.