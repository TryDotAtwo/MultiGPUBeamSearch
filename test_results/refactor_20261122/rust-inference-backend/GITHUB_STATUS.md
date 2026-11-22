# Rust inference backend gate

- Tested implementation SHA: `402e655eff1ee25118c7fb1d2033d9dff5298674`.
- Installed PEP 517 wheel, outside checkout: PASS, [Actions 45002801408](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45002801408). All 33 job steps passed, including direct Rust/PyO3 policy cases, the Rust runtime policy test, old-wheel compatibility, public preparation-path routing, and the existing installed CNOT/Gloo tests.
- Rust/Python CNOT CPU/Gloo: PASS, [Actions 45002801489](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45002801489).
- Hosted acceptance source harness: PASS, [Actions 45002801790](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45002801790). Physical CUDA acceptance jobs were skipped.
- Pinned panel-v2 source compatibility: PASS, [Actions 45002806710](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45002806710).
- RED before implementation: [Actions 45001165651](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/45001165651). The first implementation test attempt at `02868b1` exposed a test-only import error; the corrected test and implementation passed on `a0fa983`, then the defensive Rust change passed on the tested SHA above.
- Scope: Rust now owns the permutation `auto`/`libtorch`/`cutlass` policy and checks before runner construction. Python/Torch supplies device observations and preserves fallback for older wheels. Search, model scoring, kernels, precision, collectives, and result formats are unchanged.
- GPU status: NOT RUN / SKIPPED. No physical CUDA/NCCL, trained checkpoint, or 1/2-GPU parity result is claimed.
