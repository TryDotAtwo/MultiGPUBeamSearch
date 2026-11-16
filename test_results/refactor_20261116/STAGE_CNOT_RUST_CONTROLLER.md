# CNOT Rust controller and C++ frontier stages — 2026-11-16

## Scope

The tested source is Git base `1f8e40b24e3163b27c4fcd621dc18ac0073ff967`
with the pinned overlays in `cnot-rust-controller/cnot-rust-evidence.tar.gz`.
The remote job verified 556 source files before and after execution. The local
`cnot-rust-controller/verify_results.py` checks the archive hash, current
overlay bytes and the bounded acceptance receipts. The evidence archive's
SHA-256 is `16a71d8dee60d443e0ad33c686d284a3e37bd0883265c147d2de137314667b12`.

Rust now controls the CNOT search loop, stage order, termination, rank history
and path reconstruction. The caller's Python thread, process group, rank-local
CUDA device and stream are preserved. Python continues to provide scorer,
global selection, tensor and collective stages; this is not yet the complete
Rust/C++ rewrite. The Rust loop transfers no GPU address between processes.
The CUDA path packs and unpacks selected frontiers through the same ATen
operations in C++; an older externally supplied operator library falls back to
the original operations. Matrix equality, ranking, algorithms and numerical
policy were not changed. `engine="auto"` selects Rust only if installed;
explicit Python and Rust choices are available for comparison.

## Evidence

- Rust 1.100.0: `cargo test --workspace --locked` passed, with the existing
  ignored tests remaining ignored. The first attempt exposed a Python-list
  versus PyO3-tuple history error; the corrected run passed.
- Linux, PyTorch 2.9.0+cu128, CUDA 12.8, two RTX 3060 12 GiB, driver
  595.91.07: 31 focused CPU/Gloo tests passed. Eight CUDA packet/view tests
  covered n=4 and 65, empty winners and empty destination ranks. The full
  Python integration suite reported **717 passed, 10 skipped**.
- One- and two-GPU searches compared complete audited results of the old
  and Rust controllers for Qhat, proposal, full alphabet and a 144-wire case.
  The latter replayed physical IDs `[18592, 428]`. In the two-rank synthetic
  test, a scorer failure reached both ranks and a subsequent search reused
  the group successfully. These checks use the existing NCCL transport.
- With the existing native checkpoint scorer, one- and two-GPU Qhat searches
  returned the same audited result and independently replayed `[5, 7, 0]`.
  In proposal mode this small checkpoint rejected feature F02 outside its
  coverage; both controllers returned the same rejection on all ranks.
  Checkpoint SHA-256:
  `92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4`.
  Private checkpoint and provider sources are not in the evidence archive.

The fixture does not establish multi-node or Slurm behavior, other GPU
architectures, all checkpoint domains, performance, cancellation during a
collective, compatibility with an actual older external operator binary, or
completed migration of the remaining Python stages.

## Cost and cleanup

The task-owned lease `60928262` was released; a subsequent provider inventory
confirmed absence (`cnot-rust-controller/release.json`). The provider invoice
query showed $0.094 for this lease at query time, including listed GPU, storage
and traffic entries. Provider charges can lag, and this is not a final invoice
or a new total for the complete project.
