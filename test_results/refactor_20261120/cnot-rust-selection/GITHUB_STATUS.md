# CNOT Rust compatibility selection dispatch

**Scope.** For typed `PyCnotSearchInput`, Rust compatibility dispatch now calls
the existing `cnot_native_selection::select` implementation directly and
stores its returned rank/local/lexical-index tensors on the stage provider.
The Python `SelectionStages` wrapper is bypassed. Python-only input and older
stage providers retain their callback fallback.

## Hosted evidence

- RED, [run 44618218529](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44618218529),
  failed at the forbidden `CnotStages.prepare_selection` callback for qhat and
  proposal. The run reported 2 failed, 230 passed, 2 skipped.
  Artifact [14576716078](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44618218529/artifacts/14576716078),
  SHA256 `4f540d712ba29f03b084638097b00f3efd6ce858de65e81e4c43cf8c01ebe2a3`.
- Run [44618613883](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44618613883)
  stopped at rustfmt before compilation or tests.
- Run [44618983198](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44618983198)
  still reached the forbidden callback because the initial guard checked
  output attributes before selection created them. The guard was corrected.
- Branch GREEN, [run 44619334509](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44619334509),
  at code commit `4afd38f9911bdd243b8e65f4e0d789928626fff5`: Rust checks,
  PyO3 build, and CPU compatibility suite succeeded; 232 passed, 2 skipped.
  Artifact [14575149896](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44619334509/artifacts/14575149896),
  SHA256 `eca984b59d47212b4ba5f0265277c10d4048509c9f9e62c11b818c34d42896ab`.
- Main GREEN, [run 44619796548](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44619796548),
  at the same code commit: Rust checks, PyO3 build, and CPU compatibility suite
  succeeded; 232 passed, 2 skipped.
  Artifact [14575103927](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44619796548/artifacts/14575103927),
  SHA256 `25aa5f670c65f98d86a40cfa56dcc3926bac67e75e7a1cf8a6ab54f5c2b54bd2`.

The workflow wrote full logs under
`test_results/refactor_20261120/cnot-rust-selection`; GitHub Actions artifacts
retain them. These are hosted CPU/PyO3 results. Physical CUDA/NCCL multi-GPU
behavior remains unverified.
