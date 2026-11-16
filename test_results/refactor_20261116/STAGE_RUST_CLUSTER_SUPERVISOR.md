# Rust supervisor for external ranks — 2026-11-16

The tested tree is Git base `448989653d8e910375c6a9f2856adc59c09549db` plus
the SHA-256-pinned changes in `rust-cluster/rust-cluster-results/source-manifest.json`.
The source was checked before and after the initial test job. The independent
native-search fixture and the CUTLASS source archive have their own SHA-256
values in `rust-cluster-native/fixture-hashes.json` and successful remote
checks in `rust-cluster-native/rust-cluster-native-evidence/inputs.txt`.

## Implementation boundary

`run_cluster_rank(..., engine="rust")` and `cluster_worker --engine rust`
use Rust for externally assigned rank validation, process ownership, exit
status, timeout, `FAILED` marker and cancellation/reaping. Python resolves
paths and the executable before the coarse PyO3 call. The previous Python
supervisor stays the default. The native `production_runner` still owns the
CUDA work, NCCL communicator, search algorithm and global beam.

## Evidence

- Linux, CUDA 12.8 / PyTorch 2.9.0+cu128; two 12 GiB RTX 3060 GPUs, driver
  595.91.07. Source-pinned Rust `cargo test --workspace --locked` passed;
  two tests requiring a separate native fixture remained ignored.
- Focused comparison: 37 Python/Rust tests passed, including torchrun and
  simulated Slurm assignments, shared NCCL path, rank output isolation,
  stale rank collision, peer error, timeout and SIGTERM child cleanup.
  The full Python integration suite reported 698 passed and 12 skipped.
- Both supervisors launched a two-rank torch.distributed NCCL all-reduce;
  both ranks reported world size 2 and sum 3, and saw the same NCCL ID path.
  This is a transport smoke test, not a production solver result.
- The production CUDA/CUTLASS runner was then built from the pinned source and
  tested through both supervisors with the same untrained Hamming MLP, a 96-byte
  permutation problem, two GPUs, depth limit 4 and global beam 64. Both returned
  `[0, 1, 2]`; the fixture independently replayed each solution. The runner
  SHA-256 was `ff0eea84c384e6bdb1321a9615cf37e63130f9e85566263cd059dff0f763a2c0`.
  See `rust-cluster-native/rust-cluster-native-evidence/search/result.json`.

This is one node with two GPUs. Simulated Slurm environment variables are not
a real Slurm allocation. Multi-node networking, rank failures on real multi-node
NCCL, other GPU families, CNOT search and final default-engine migration remain
unverified. No performance benchmark or inference-policy change was performed.
The experimental PyO3 adapter currently expects UTF-8-compatible argument and
environment strings; it is not a complete replacement for all Python/OS byte
representations. The downloaded native evidence omits the copied NCCL runtime
binary, retaining its SHA-256 and the full original archive SHA-256 instead.
