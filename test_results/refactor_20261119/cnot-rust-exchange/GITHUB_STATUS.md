# Rust CNOT candidate exchange dispatch

Scope: move only typed compatibility-backend exchange dispatch to Rust while
reusing the existing exchange helper and preserving the Python fallback.

## Implementation

For typed `PyCnotSearchInput` with `world`, `candidates` and `owners`,
`PyStages::exchange_candidates` calls
`cnot_native_exchange::exchange(..., native_packet=true)` and stores the
returned tensor as `incoming`. The helper still controls the single-rank
short-circuit, optional CUDA packet path, Stream5 transport and
`all_to_all_single` fallback. Python-only or older stage providers still call
`CnotStages.exchange_candidates`.

No search logic, payload layout, owner routing, rank topology, collective
ordering, transport protocol or kernel changed.

## Hosted verification

- RED: [Actions run 44596999245](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44596999245) failed at the expected Python `CnotStages.exchange_candidates` callback: 1 failed, 221 passed, 2 skipped. Artifact: 14566162577.
- Formatting-only: [Actions run 44597579671](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44597579671) stopped at `rustfmt --check`; compilation and tests did not run. Artifact: 14565369738.
- GREEN: [Actions run 44597954192](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44597954192) passed Rust formatting/checks, the PyO3 build, and CPU compatibility tests: 222 passed, 2 skipped. Artifact: 14565748082.
- The GREEN check is CPU/PyO3 evidence. It does not verify physical CUDA/NCCL or multi-GPU runtime behavior.
- MAIN verification after fast-forward: [Actions run 44598940460](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44598940460) passed the same checks (222 passed, 2 skipped; artifact 14566448047).

Full logs remain in GitHub Actions artifacts. No local builds, runtime dependencies, logs or test outputs were created on Windows.
