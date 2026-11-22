# CNOT Rust compatibility tensor fallback dispatch

**Scope.** Typed `PyCnotSearchInput` compatibility expansion and
distinct/scoring no longer call the high-level Python helpers `cnot._expand`,
`cnot._distinct`, or `cnot._scores`. The Rust adapter uses the existing
`cnot_native_batch` tensor path when fused CUDA operators are unavailable.
Existing fused CUDA hooks remain preferred.

Lower-level CNOT domain primitives remain the existing `cnot_native.py`
interface. CUDA inputs use the existing compiled operators where available;
CPU/reference execution still uses Torch. No CUDA kernel, scoring precision,
search algorithm, state layout, or transport changed.

## Hosted evidence

- RED, [run 44621164737](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44621164737),
  hit the forbidden `cnot._expand` and `cnot._distinct` helpers for qhat and
  proposal. The run reported 4 failed, 232 passed, 2 skipped.
  Artifact [14576679253](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44621164737/artifacts/14576679253),
  SHA256 `96799f794f77e29c0dfb09e1cb6a3dfc438406f019cd2a266de58963661f98bf`.
- Branch GREEN, [run 44622029810](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44622029810),
  at code commit `ef0ca2ad35f729f15a492b39bc91f504ff6a0725`: Rust checks,
  PyO3 build and CPU compatibility suite succeeded; 236 passed, 2 skipped.
  Artifact [14578940035](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44622029810/artifacts/14578940035),
  SHA256 `f7e237eae76057521ec0d48cca544c231abde09cfe5c8ed7810e761ccd80b0f3`.
- Main GREEN, [run 44622384926](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44622384926),
  at the same code commit: Rust checks, PyO3 build and CPU compatibility suite
  succeeded; 236 passed, 2 skipped.
  Artifact [14575724693](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44622384926/artifacts/14575724693),
  SHA256 `f80dd0f54adb6e497bfc7b762ec37c7f63df71d4a94054493fea56d521c2c00f`.

The workflow wrote full logs under
`test_results/refactor_20261120/cnot-rust-domain-fallback` and retained them in
Actions artifacts. This CPU/PyO3 suite does not verify physical CUDA/NCCL
multi-GPU behavior.
