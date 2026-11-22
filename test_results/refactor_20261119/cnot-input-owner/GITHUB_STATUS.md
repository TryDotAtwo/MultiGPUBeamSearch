# Rust-owned CNOT search input — verification record

- Test-first run `44407881591` failed for the expected missing `CnotSearchInput` type: 1 failed, 162 passed, 2 skipped.
- Green branch run `44408663743` passed Rust checks, PyO3 build, CNOT CPU compatibility tests (163 passed, 2 skipped), and NCCL probe syntax.
- Hosted artifact `14476796556`, SHA-256 `2f41fd59fbca781ad91269f8731f06544af62c0a7ba99bea15329eab5424d8ec`.
- Matching extensions now return a frozen `CnotSearchInput` carrying the validated `beam_core::cnot::CnotMatrix`, the same detached tensor snapshot, and raw bytes. The Python adapter passes this object through; four-item results from older extensions remain supported.
- Published main revision `6db529828b23df978248838f81e7ca183a0388ea` passed CNOT CPU workflow `44409374208` (163 passed, 2 skipped); artifact `14475029346`, SHA-256 `7a7cb686d19de9292d54324e4f2795136287cbe1b63c4c82413a46d8fdaf850b`.
- Search, scorer, and collective behavior did not change. Physical CUDA/NCCL or multi-GPU CNOT execution remains unverified.