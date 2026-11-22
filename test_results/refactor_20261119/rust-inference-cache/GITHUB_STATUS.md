# Rust capacity-cache decision — verification

- Test-first RED commit: `6aa3c093ed87f963b4368d48360192c5dc4d1344`; the first Actions run did not execute the added file because the explicit pytest list omitted it. The workflow command was corrected in `07834e0ca1572276f5999e979524677f3c331ffb`.
- Verified RED: Actions run [44549277542](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44549277542), 9 new expected failures, 202 passed, 2 skipped.
- Implementation: `a155a98735abeb323eb825baf8360ed50da37360`; rustfmt correction: `8dc4092e0457c372c14882893d7db70898aaab92`.
- GREEN: Actions run [44550275840](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44550275840), success; 211 Python tests passed, 2 skipped, scoped Rust groups passed, PyO3 build/check passed, and NCCL probe syntax passed.
- Artifact: [cnot-compat-root-goal-cpu-validation, ID 14543596077](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44550275840/artifacts/14543596077), SHA-256 `8a4bea305bd3cdbaa0a00ceb2b3a37c5cfcc4214c1e15a542301d13f44903fd1`.
- Rust compares UUID-to-MiB maps with the existing key-equality and 32 MiB pressure rule. Python still performs `nvidia-smi` memory sampling and retains the non-capacity short-circuit plus fallback.
- No algorithm, cache schema, GPU kernel, or multi-GPU protocol changed.
- Limits: CPU/PyO3 only. NCCL probe syntax is not a collective run. No physical GPU or Vast compute was used.
