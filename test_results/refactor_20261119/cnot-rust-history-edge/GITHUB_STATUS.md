# Typed CNOT history-edge dispatch — GitHub evidence

Scope: On the typed Rust CNOT path, the compatibility backend copies the owner-held history record in Rust, calls the same `World.broadcast` operation, then validates the parent, route and owner. Python-only inputs and older providers keep the existing callback. History layout, rank ownership, collective order and path semantics are unchanged.

- RED: [Actions run 44592673898](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44592673898), 1 failed, 220 passed, 2 skipped; the expected failure was the Python `CnotStages.history_edge` callback. Artifact ID: 14563163386.
- Formatting-only failure: [Actions run 44593184432](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44593184432) stopped at `rustfmt --check`; no Rust or Python tests ran. Formatting was corrected.
- GREEN: [Actions run 44593458330](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44593458330), Rust checks and PyO3 build passed; selected CPU suite: 221 passed, 2 skipped. Artifact ID: 14565280008.
- The regression search uses a two-gate input and verifies the reconstructed path by replay. No physical CUDA/NCCL or multi-GPU run occurred.

Full logs remain in GitHub Actions artifacts. No local builds, results, or logs were created on Windows.
