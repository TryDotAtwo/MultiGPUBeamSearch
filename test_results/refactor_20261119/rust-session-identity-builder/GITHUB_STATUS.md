# Rust resident SessionIdentity builder

## Scope

The optional Rust/PyO3 builder assembles and serializes the typed launch `SessionIdentity`. `RustResidentWorkers` uses it when present; older extensions retain the previous Python JSON payload. The identity is built once per resident configuration and reused while that configuration remains unchanged. Search behavior, ranking, numerical policy, rank order, CUDA stages, and NCCL protocol are unchanged.

## RED verification

- Test-only commit: `420d39f47d35bc64f214e45e068af0f0097f3984`.
- The test-only runs below recorded the RED phase before implementation. After implementation, runs `44530744337` and `44530743911` exposed a test expectation error: the test expected three identity builds, but only two resident configurations were created (initial and changed model). A compatible repeated session correctly reuses its identity.
- Python owner run: [44524014350](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44524014350); CNOT CPU run: [44524013917](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44524013917). The later implementation runs [44530744337](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44530744337) and [44530743911](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44530743911) reported the same assertion failure.

## GREEN verification

- Corrected test commit: `638bcab144a8b10a9d97e82a8246e24d1ac59ecd` on `main`. The test checks two builds and confirms the second contains the changed model artifact hash.
- Python owner CPU run [44535162475](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44535162475): 83 passed, 3 skipped. Artifact [14535731012](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44535162475/artifacts/14535731012), SHA-256 `c8ab33c4639473beeb786b669f07117311eb625dcb5af6e3fbe7612550fd87e0`.
- CNOT CPU run [44535162234](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44535162234): Rust CNOT checks passed, PyO3 extension built, 180 passed and 2 skipped, NCCL probe syntax check passed. Artifact [14534959024](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44535162234/artifacts/14534959024), SHA-256 `4acdd2e0b002b83dfe4c86fbf841ffe98d86bc9b29ee8e92871e57ffa2ed3a86`.

## Limits

No local build or test was run. These CPU checks and NCCL probe syntax validation do not establish physical CUDA/NCCL execution or multi-GPU CNOT behavior.
