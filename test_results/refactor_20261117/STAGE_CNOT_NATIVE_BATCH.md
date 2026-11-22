# Rust-owned CNOT batch orchestration

The Rust CNOT stage backend now performs score validation and orchestrates the
existing CPU/older-CUDA expansion and distinct-child tensor operations. The
current CUDA ATen operators, Torch storage, scorer callbacks, full-matrix
equality, rank routing, collective schedule and numerical policy are retained.
The compatible older Python stage adapter remains available.

The final green evidence uses base
`ecb675a68f6a6c891b9c1e165b899e225e9e4024` plus five SHA-256 source overlays.
It builds in a fresh Cargo target and verifies every source file before and
after execution. Archive SHA-256:
`a82b5a3590fccbc66f398f0113e7b7e97cd9eb9b7e5e1dc45be51868c3f795f1`.

- Rust workspace checks, 91 focused CPU/Gloo/CUDA-controller tests, 62 CUDA
  tests and the full Python suite passed: 831 passed and 10 skipped.
- Five audited one/two-GPU CNOT searches matched the Python reference while
  Python score, expansion and distinct helpers were forbidden. On one/two
  RTX 3060 GPUs, older expansion and distinct operators were simulated alone,
  together and with different missing operators on different ranks.
- Gloo tests check score shape, dtype, device and finite-value errors against
  the Python reference. Existing NCCL failure/recovery, Stream 5, radix, goal
  and checkpoint-scorer one/two-GPU checks also passed.
- The pre-implementation red archive SHA-256 is
  `de2d4046d115d0bee99a35ba32d9d5f373529e630aaa04e597135646f31a1161`:
  three searches failed as expected with the old Python helper dependency.
  The fresh-target smoke passed 28 tests; its archive SHA-256 is
  `3e1bddcddaa036bb3d58b8b09ea2e40bf516a36b7d02d49fef7b8a8ee0090699`.
- `cnot-native-batch-green/verify_results.py` checks pinned sources, archived
  results, rank receipts and provider lease-release receipt. Instance
  `60988541` was deleted and provider absence verified. Its last reported
  duration and rate imply approximately $0.024; final billing may differ.

This is an intermediate migration. Python is still involved in admission,
scorer callbacks, Torch domain primitives and parts of collective management;
the Rust/C++ CNOT runtime is not yet complete. The hardware evidence covers
one host with one/two RTX 3060 GPUs. Multi-node/Slurm and physical T4, Hopper
and Blackwell acceptance remain open. Compute Sanitizer was not rerun for
this host-orchestration stage; the earlier filtered result and its limits are
recorded in `STAGE_CNOT_NATIVE_GOAL.md`.
