# Rust preparation result metadata reuse

## Source

- Test-only RED commit: [d190772](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/d1907724b0861a0de1d3383cc81202a8bc39de22).
- Implementation: [0581fb5](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/0581fb5a83c9885f452f50316c6476c32be1977a).
- RED run [44484788342](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44484788342) found the missing Rust metadata owner API (3 failed, 76 passed, 1 skipped); run [44484788323](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44484788323) stopped at the expected missing Rust method during compilation.

## GREEN hosted checks

- [Rust public owner, run 44486326212](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44486326212): 79 passed, 1 skipped. Artifact [14511001662](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44486326212/artifacts/14511001662), SHA-256 6dd709c3688beeb623abd46e5176c2ea1e2a6350d0da974a04ee4b3f25c1c9ef.
- [Permutation depth control, run 44486326192](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44486326192): Rust test groups 5 and 2 passed; formatting, cargo checks and CPU native ABI fixture passed. Artifact [14510312919](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44486326192/artifacts/14510312919), SHA-256 e459c17102cb6572f2a65740530a6725212cb2500062f5789b541346e040372f.
- [CNOT CPU compatibility, run 44486326225](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44486326225): Rust formatting; beam-core CNOT/search 5 and 9 passed; runtime CNOT 4 passed; PyO3 build and checks passed; Python compatibility 174 passed, 2 skipped; NCCL probe syntax passed. Artifact [14510922042](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44486326225/artifacts/14510922042), SHA-256 5c89bd2807124fb3f81f8fc980e8023d1275f29834aa445287f72644cf2e99d0.

Full job output and test logs are retained in the Actions artifacts above. No physical CUDA/NCCL collective or multi-GPU acceptance run occurred, and no Vast instance was started.

## Scope

Rust owns and refreshes the full prepared metadata for compatible resident sessions. The PyO3 boundary remains coarse and serializes one result per search; no calls were added per candidate or CUDA kernel. Older extensions retain the Python metadata-cache/merge path. Search ordering, CUDA kernels, rank exchange and public Python result formats are unchanged.
