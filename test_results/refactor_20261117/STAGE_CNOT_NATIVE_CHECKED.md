# Rust-owned CNOT stage error coordination

The Rust CNOT stage backend now performs the existing rank-wide error vote and
failure-only message gather for initialization, expansion and distinct-child
evaluation. It preserves the operation order, original single-rank exception,
rank-ordered messages and the existing PyTorch/NCCL collectives. The legacy
Python stage backend and Python admission checks remain available. Torch still
owns all tensor storage, and no GPU pointers or numerical operations changed.

The final green evidence uses base
`1fa662aa12b4c5676613c3c2e74ecba2de5593f1` plus seven SHA-256 source overlays.
It builds in a fresh Cargo target, checks the new extension entry, and verifies
the source files before and after execution. Archive SHA-256:
`eb1b2a57f0b3ea6d34c4635dc45671cdcb0372c8ca8886f0a102bd01108f2dd6`.

- Rust workspace checks, 80 focused CPU/Gloo/CUDA-controller tests, 62
  additional CUDA tests and the full Python suite passed. The latter reported
  820 passed and 10 skipped on Linux, CUDA 12.8, PyTorch 2.9.0 and two RTX 3060
  GPUs.
- On one/two GPUs, five audited searches matched the Python reference while
  forbidding Python's goal and stage-error coordinators for the Rust backend.
  The compatible Python stage backend was retained collectively when one rank
  lacked the newer interface.
- Gloo and NCCL tests injected failures on rank 0, rank 1 and both ranks,
  checked identical rank-ordered errors and reused the process group. They also
  checked the single-rank original exception and `BaseException` behavior.
- Existing Stream 5, CNOT radix/goal and checkpoint-scorer one/two-GPU tests
  passed. This stage changes host-side coordination and does not change CUDA
  memory operations; Compute Sanitizer was not rerun for this commit. The
  immediately preceding goal stage's filtered memcheck and its limitations
  are recorded in `STAGE_CNOT_NATIVE_GOAL.md`.
- The pre-implementation red archive SHA-256 is
  `f75e7c236a7d211df01b1aed88e90835a2edc203515c814104f7f0fd4df57b55`;
  the new tests failed as expected against the base version. The initial
  fresh-target smoke passed 23 tests and its archive SHA-256 is
  `ccbd47bfe18813ca9779f491a59e556550f96bebf87ff5bc6fbfbea22f961717`.
- `cnot-native-checked-green/verify_results.py` validates the pinned sources,
  archived tests, rank receipts and provider lease-release receipt. Instance
  `60986684` was deleted and provider absence verified. Its last reported
  duration and rate imply approximately $0.031; final billing may differ.

This is an intermediate migration. Python is still involved in admission,
user scorer calls, some tensor stages and collectives, and old-library
fallbacks. The hardware evidence covers one host with one/two RTX 3060 GPUs;
multi-node/Slurm and physical T4, Hopper and Blackwell acceptance remain open.
