# Rust CNOT construction verification dispatch

Scope: bypass only the Python path-verification callback for typed CNOT search
input, using the already-owned Rust matrix verifier.

## Implementation

For typed `PyCnotSearchInput`, `PyStages::verifies_construction` calls
`search_input_verifies_construction`. The existing Rust `CnotMatrix`
validates the reconstructed gate sequence. Python-only inputs and older stage
providers keep `CnotStages.verifies_construction`.

No path reconstruction, gate policy, rank ownership, scorer, or search
semantics changed.

## Hosted verification

- RED: [Actions run 44600364785](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44600364785) failed at the expected Python `CnotStages.verifies_construction` callback: 1 failed, 222 passed, 2 skipped. Artifact: 14567308560.
- GREEN: [Actions run 44600777706](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44600777706) passed Rust formatting/checks, the PyO3 build, and CPU compatibility tests: 223 passed, 2 skipped. Artifact: 14566579277.
- This is CPU/PyO3 evidence. Physical CUDA/NCCL and multi-GPU runtime behavior were not tested.

Full logs remain in GitHub Actions artifacts. No local builds, runtime dependencies, logs or test outputs were created on Windows.
