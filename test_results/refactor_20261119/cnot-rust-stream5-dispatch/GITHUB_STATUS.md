# Rust-owned CNOT Stream 5 dispatch: GitHub verification

## Change
- Rust CNOT controller calls the existing Rust Stream 5 setup helper directly.
- The helper still chooses between the same Rust and Python transport classes, performs the same rank-wide capability vote, and assigns the same transport object to `world.transport`.
- Python `World.start_transport` remains available for the old-extension compatibility path. Search ordering, rank topology, Stream 5 transport, CNOT scoring and equality rules are unchanged.

## Red test
- Action [44580579687](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44580579687), commit `fb50554a55687b0f4f175a56789504d052dd3621): expected failure at `World.start_transport`, which the Rust top-level controller still called before this change. Other selected checks: 218 passed, 2 skipped.
- Remote Actions artifact: [14559500020](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44580579687/artifacts/14559500020), SHA-256 `cb3979ede479c24d752670672434624f94a70d7b96ec73683f7174b1bd5dea74`.

## Green test
- Action [44581088466](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44581088466), commit `39cf0ef67c944589f70dedde3bb22c3f25be3c10`: Rust formatting/checks and selected Cargo tests passed; PyO3 extension built; Python compatibility suite passed (219 passed, 2 skipped); the NCCL probe passed `py_compile`.
- Full remote log and workflow outputs: artifact [14559050896](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44581088466/artifacts/14559050896), SHA-256 `c74344cd0f026fa31d2ae70264023d438c86649f551a2e7e174073f5814d7ba8`.

## Limits
- The probe was syntax-checked only. No physical CUDA or multi-GPU NCCL runtime, Vast instance, or paid compute was used.
