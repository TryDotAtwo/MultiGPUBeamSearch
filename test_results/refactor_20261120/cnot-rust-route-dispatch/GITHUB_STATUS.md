# CNOT route packing dispatch moved into Rust

**Scope.** Typed PyCnotSearchInput compatibility expansion now validates the
source rank and move tensor in Rust, applies the existing uint16 range check,
and builds the same int64 route lane. It no longer calls
cnot_native.pack_cnot_route.

The child generation, owner assignment, distinct-state primitives, score
calculation, tensor layout, stage order, and collectives are unchanged.
cnot_children, cnot_first_unseen, and cnot_owners remain existing C++/CUDA or
CPU/Torch domain operations. No new kernel or search optimization was added.

## Hosted evidence

- Run [44625061344](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44625061344)
  is not RED evidence: its explicit pytest list omitted the new test.
- RED, [run 44625601360](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44625601360),
  exercised qhat and proposal and failed at the forbidden Python route helper
  (2 failed, 236 passed, 2 skipped). Artifact
  [14578637524](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44625601360/artifacts/14578637524),
  SHA256 c1b408b8c353228f22e2cc60a12f56ae3345a6acfe3fe5acfefe7881350c9d0e.
- Formatting-only run [44626077505](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44626077505)
  stopped at rustfmt before compilation or tests.
- Branch GREEN, [run 44627095448](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44627095448),
  at code commit 6f1d72955b04ade46a28c43b1e1310bb56e5ecd0: Rust checks,
  PyO3 build, and CNOT CPU suite passed (238 passed, 2 skipped); the
  two-rank NCCL probe was syntax-checked only. Artifact
  [14579484694](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44627095448/artifacts/14579484694),
  SHA256 9a420188f9dd7eed2c20bce4ae48585ecc9f639330e7190a7d7a3eaa7028db65.

Physical CUDA execution, NCCL collectives on GPUs, and multi-GPU CNOT runtime
remain unverified. Full logs are retained in the GitHub Actions artifact.