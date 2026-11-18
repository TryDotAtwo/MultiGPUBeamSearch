# CNOT optional extension import fallback — hosted CPU evidence

- Code under test: `bae5697ae386dc7cd2a1d118c6cd84cfcaf49e2f`.
- Workflow: [CNOT CPU compatibility run 44227654569](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44227654569) — success.
- Checks: Rust core CNOT/search, runtime CNOT, native-sys TQP scorer, PyO3 `cargo check` and build; 122 Python tests; two-rank NCCL probe syntax.
- Regression: auto validation and engine fall back when both supported Rust module names fail to import. Explicit Rust validation/engine still fail closed.
- Artifact: [14393946014](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44227654569/artifacts/14393946014), SHA-256 `6481aa6e633831229d7169fa1464102c17f07107d4148cb4e77f7a62c7608726`.
- Red run: 44227238318 failed as expected on the new case (1 failed, 121 passed).
- Not verified: CUDA execution, NCCL collectives at runtime, physical multi-GPU behavior, Compute Sanitizer, performance, Kaggle runtime.
