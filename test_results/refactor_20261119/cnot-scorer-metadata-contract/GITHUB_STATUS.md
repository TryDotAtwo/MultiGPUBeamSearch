# CNOT checkpoint scorer metadata: hosted verification

Date: 2026-11-19

## Scope

Rust `beam-native-sys::TqpScorer` now validates the JSON metadata contract (102 feature inputs and IDs, integer `max_packet_rows`) during open. Rust chooses the effective inference batch. PyO3 preserves the public metadata dictionary and existing scorer interface.

## Results

- RED: [GitHub Actions run 44569025127](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44569025127) compiled the added tests and failed at the expected missing `CnotScorerMetadata` implementation.
- GREEN: [GitHub Actions run 44570920822](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44570920822) passed rustfmt, scoped Rust checks, all 8 native-sys scorer tests, PyO3 extension build, 219 Python tests (2 skipped), and NCCL probe syntax.
- Hosted artifact: [14553567932](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44570920822/artifacts/14553567932), SHA-256 `d24e4df850867c6f291ff6bb736eac3a4b2fc3846ed6f50c1ec1b8c11737bc54`.
- This did not run a physical GPU scorer or a multi-GPU NCCL search.

