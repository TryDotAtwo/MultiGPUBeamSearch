# Rust-owned CNOT admission collectives: GitHub verification

## Change
- The Rust CNOT top-level controller calls the existing Rust consensus helper directly.
- It also calls the existing Rust all-rank native-stage support vote directly.
- The same Python process-group primitives, Stream 5 startup adapter, fallback stage path, rank order, CNOT equality/scoring rules, and public API remain in place.

## Red test
- Action [44576411236](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44576411236), commit `caa7a60d0a248a8eb149e2b739dac3325f5c4dbc): expected failure at `World.require_consensus`, which the Rust controller still called before the migration. Other selected checks: 218 passed, 2 skipped.

## Intermediate test-double correction
- Action [44577310497](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44577310497) compiled Rust and PyO3 but failed one compatibility-fallback test because its fake `sum` returned 1 for the zero-valued admission error flag. The test double now models the actual reduction with `tensor.sum()`.

## Final green test
- Action [44577927456](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44577927456), commit `c6779820b2fed79a554b2c470bd646697c08f077`: Rust formatting/checks and selected Cargo tests passed; PyO3 extension built; Python compatibility suite passed (219 passed, 2 skipped); the NCCL probe passed `py_compile`.
- Full remote log and workflow outputs are in Actions artifact [14556663231](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44577927456/artifacts/14556663231) (SHA-256 `e78d64e5ed298260ac113b20f0cb6b3b419ff56a0b1cbdbbc6a20f76b2ca5b6d`).

## Limits
- The NCCL probe was syntax-checked only. No CUDA execution, multi-GPU collective runtime, Vast instance, or paid compute was used.
