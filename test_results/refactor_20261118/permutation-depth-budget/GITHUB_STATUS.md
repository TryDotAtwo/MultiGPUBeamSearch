# Rust permutation depth-budget control

Latest source commit: `5517288e9215f383b081bcbff4b80e6634031ba5`.

- Hosted workflow [44216092965](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44216092965) passed Rust formatting, 3 `beam-core` CNOT tests, 4 `beam-core` search tests, 4 `beam-runtime` CNOT tests, 4 `beam-native-sys` scorer tests, `cargo check -p beam-python`, the PyO3 extension build, 38 Python tests, and NCCL probe syntax compilation.
- Full log artifact [14388107367](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44216092965/artifacts/14388107367), SHA-256 `0ed3aa03c26239652506b4a6cab4896b74a15eaf031123c7b1401b1c172cbfee`.
- The first run [44215569290](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44215569290) passed Rust/PyO3 checks and 36 Python tests, while the two new oversized-integer cases failed because the wrapper called a narrower Rust signature. The follow-up preserved Python arithmetic outside `u64`/`u32` ranges; the successful run above includes both regressions.
- These checks cover CPU arithmetic, extension integration, and compatibility only. No CUDA scoring, NCCL communication, or physical multi-GPU search ran.
