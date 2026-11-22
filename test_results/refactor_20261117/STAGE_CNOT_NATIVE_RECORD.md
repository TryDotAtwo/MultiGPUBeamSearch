# Rust-owned CNOT record exchange orchestration

The Rust CNOT stage backend now prepares and dispatches rank record exchanges.
It preserves the same stable C++ ATen packet operator or Torch fallback, the
selected existing Stream 5 NCCL transport, the CPU/Gloo all-to-all sequence,
empty splits, rank ordering and Torch tensor ownership. Single-rank calls return
the original tensor. The Python reference and older stage adapter remain usable.

The final green evidence uses base
`4a16113cd376ed99fcdda2afa341d3270e6d78df` plus seven SHA-256 source overlays.
It builds in a fresh Cargo target and verifies every source before and after
execution. Archive SHA-256:
`a87983773e11cf73a8227c7502d80a8e633a15cd06521991bf120f156459b3cb`.

- Rust workspace checks, 95 focused CPU/Gloo/CUDA-controller tests, 62 CUDA
  tests and the full Python suite passed: 835 passed and 10 skipped.
- One/two-GPU audited searches matched the Python reference with the legacy
  Python record exchange forbidden for the Rust stage backend. Two-rank Gloo
  checks exercised stable record order, one or both ranks empty and reuse.
- Two-rank NCCL checked three packet-operator availability scenarios (native,
  simulated fallback on both ranks and mixed rank versions) with four split patterns,
  including a delayed rank. Existing goal, radix, Stream 5, coordinated-error
  and one/two-GPU checkpoint-scorer checks passed.
- The pre-implementation red archive SHA-256 is
  `0ab959bdc9ab236dc9da2da1e90f44c26267b55cf4138b3a9ae70bca42ec610f`:
  three tests failed as expected. The fresh-target smoke passed 32 tests;
  its archive SHA-256 is
  `672c19ce8a8a8f05c1add9f44593cb09d82854b4e6b7c1b4f4107187db333fec`.
- `cnot-native-record-green/verify_results.py` checks pinned sources, archived
  results and provider lease-release receipts. Instance `60990077` was
  deleted and provider absence verified. Its last reported duration and rate
  imply approximately $0.033; final billing may differ.
  An earlier cheaper instance (`60989817`) did not accept SSH and was deleted;
  fresh provider inventory verified absence. The first release receipt could
  not be written because its destination directory did not yet exist, so the
  subsequent absence receipt has no duration or price snapshot.

This is an intermediate migration. Admission, scorer callbacks, Torch domain
primitives and parts of collective management still involve Python; this is
not a complete Rust/C++ CNOT runtime. Hardware evidence covers one host with
one/two RTX 3060 GPUs. Multi-node/Slurm and physical T4, Hopper and Blackwell
acceptance remain open. Compute Sanitizer was not repeated for this host-side
orchestration stage; earlier filtered evidence and its limits are recorded in
`STAGE_CNOT_NATIVE_GOAL.md`.
