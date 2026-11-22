# CNOT entered-World validation

Date: 2026-11-20
Main commit: [8e494c5b0ac0cdedf44c21eb2cc459c466132a2b](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/8e494c5b0ac0cdedf44c21eb2cc459c466132a2b)

## Scope

The CNOT Gloo helpers now create and destroy their control group through `with World(...)`. The compatibility-stage goal tests use a real `World` instance and exercise its optional ABI capability vote. Search, scoring, selection, ownership, record exchange, and numerical behavior are unchanged.

## Hosted verification

- RED: [run 44714626102](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44714626102): 4 failed, 274 passed, 2 skipped. The failures were two test doubles missing `World` capability methods and two Gloo helpers using `World` outside its entered context. Artifact [14618627216](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44714626102/artifacts/14618627216), SHA-256 `976e92b81ffb09e27915228e1c6f2ae397e8c51664fee0648a2972bcddf59533`.
- GREEN: [run 44715949615](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44715949615): Rust CNOT checks passed, PyO3 extension built, 278 Python tests passed, 2 skipped, and the NCCL probe passed `py_compile`. Artifact [14618694452](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44715949615/artifacts/14618694452), SHA-256 `6b570f42d4244585b5abc4e015b79494daacab19b663d37a75e8ae07848a346d`.

## Limits

This validates Linux CPU/Gloo multiprocess behavior and the hosted PyO3 build. It does not validate physical CUDA/NCCL. The status-only Vast run [44714626948](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44714626948) observed instance 61080396 exited and skipped build and GPU execution; no paid compute was started.
