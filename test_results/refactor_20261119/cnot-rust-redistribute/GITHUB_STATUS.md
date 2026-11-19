# CNOT Rust compatibility redistribution

**Scope.** Rust now owns the `redistribute` stage orchestration for typed
`PyCnotSearchInput` in the compatibility backend. Existing optional CUDA
operators, the existing exchange transport, and Python fallback for older
providers remain in place. Search semantics and transport were not changed.

## Hosted evidence

- RED, [run 44613780854](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44613780854),
  failed because the compatibility backend still called the Python
  `CnotStages.redistribute` callback for both qhat and proposal. The test
  suite reported 2 failed, 228 passed, 2 skipped.
  Artifact [14574401283](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44613780854/artifacts/14574401283),
  SHA256 `b193d6bba0fd599faef7afaedcac9840...`.
- An initial implementation run, [44615323541](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44615323541),
  stopped at rustfmt before compilation or tests. Formatting was corrected.
- Branch GREEN, [run 44615705900](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44615705900),
  at tested code commit `93e09e0c68cc90165b0ba966c190b717a864b82b`:
  Rust checks, PyO3 build and CPU compatibility suite succeeded;
  230 passed, 2 skipped.
  Artifact [14573979022](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44615705900/artifacts/14573979022),
  SHA256 `be8cc17f4dd759a6b64bcd54fb87cbacbfe442a8d4170805422f0d3477c361bc`.
- Main GREEN, [run 44616358356](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44616358356),
  at `922d177e7629bea44c0f293335e2484883876dab`:
  Rust checks, PyO3 build and CPU compatibility suite succeeded;
  230 passed, 2 skipped.
  Artifact [14576135151](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44616358356/artifacts/14576135151),
  SHA256 `e761f9675a1fd90a7c46333635c4a7edf932dfc7a38b4eaa54906e154eaebbc1`.

Full runner logs are retained in the listed GitHub Actions artifacts and were
written by the workflow under `test_results/refactor_20261119/cnot-rust-redistribute`.
This is CPU/PyO3 evidence; physical CUDA/NCCL multi-GPU behavior remains
unverified.
