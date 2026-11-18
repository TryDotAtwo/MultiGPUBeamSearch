# Rust validation for normalized public search controls

- GitHub Actions run [44249581929](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44249581929) passed on `main` at `d935f57f22076b385bfa46488dab4a4dc822d1b1`.
- The workflow passed Rust formatting, `beam-core` CNOT and search tests (3 and 7 passed), Rust runtime/native-sys scorer tests, `cargo check -p beam-python`, the PyO3 extension build, and the NCCL probe syntax check. The Python compatibility suite passed 125 tests in 6.84 seconds.
- Full logs are in the [Actions artifact `cnot-tqp-scorer-cpu-validation`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44249581929/artifacts/14403152495) (artifact ID `14403152495`).
- This was a hosted CPU run using CPU PyTorch. It verified Python-to-Rust validation and CPU control flow; it did not execute CUDA kernels, NCCL collectives, or a physical multi-GPU search. Those acceptance checks remain open.
