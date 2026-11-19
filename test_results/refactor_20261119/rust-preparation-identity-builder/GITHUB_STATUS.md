# Rust preparation identity builder

## Scope

The Rust-owned `NativeSession` path now asks the PyO3 extension to construct and serialize the typed `PreparationIdentity`. If an older extension lacks this optional method, Python retains its previous JSON builder. Search semantics, rank ordering, CUDA/NCCL stages, and public result fields are unchanged.

## Test-first evidence

- Test-only revision: `bea7c3c4e8f62c947c5143db2069e8083ceef67e`.
- Rust public owner CPU run: [44509341963](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44509341963), failed.
- CNOT CPU compatibility run: [44509341981](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44509341981), failed.
- Full logs remain in GitHub Actions. They were not downloaded to the workstation.

## GREEN verification

- Rust public owner CPU: [run 44518629892](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44518629892), completed successfully.
- CNOT CPU compatibility: [run 44518629802](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44518629802), completed successfully.
- Permutation depth control CPU: [run 44518629764](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44518629764), completed successfully.
- These are hosted Actions status results; full logs remain on GitHub and were not downloaded.

## Limits

These CPU/PyO3 checks do not verify physical CUDA/NCCL execution or multi-GPU CNOT runtime.
