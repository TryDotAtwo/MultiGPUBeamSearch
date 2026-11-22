# Rust-owned CNOT goal coordination

The Rust CNOT stage backend now orchestrates the rank-local goal check and
rank-wide winner/error exchange. It calls the existing CUDA/ATen local-goal
operator when available, or the same Torch tensor operations used by the
reference. PyTorch retains tensor storage, and the legacy Python stage adapter
remains available for older extension combinations. No search ordering,
collective sequence, inference kernel or precision policy is changed.

The final green evidence uses base
`267b84d78747e1c30dc0694f8f870ec1d0cf8111` plus seven SHA-256 source
overlays. The build uses a fresh Cargo target, verifies the new extension entry,
and checks source files before and after execution. Its archive SHA-256 is
`9227d501be010c3a42b8b674d3c04fe4cb85214b663ea64f247a670d5ae58369`.

- Rust workspace checks, 72 focused CPU/Gloo/CUDA-controller tests, and 62
  additional CUDA tests passed. The full Python integration suite reported
  812 passed and 10 skipped on Linux, CUDA 12.8, PyTorch 2.9.0 and two RTX
  3060 GPUs.
- One- and two-GPU complete CNOT searches matched the Python reference in five
  audited cases, including qhat, proposal and 144 wires. Rust searches forbid
  the Python goal coordinator during these comparisons. Both ranks retain the
  older stage backend when one lacks the new extension interface.
- Goal comparisons cover empty and noncontiguous batches, goal tie ordering,
  4/65/144 wires and an empty Gloo rank. Injected errors on rank 0, rank 1,
  and both ranks reach all participants and permit process-group reuse. A
  single-rank exception and a `BaseException` retain their original type.
- The two-rank NCCL goal fixture covers injected native errors and group reuse,
  plus a mixed operator pair with the old fallback on only one rank. Existing
  Stream 5, radix, Python-goal and CNOT checkpoint-scorer fixtures also pass.
- The frozen pre-implementation red archive SHA-256 is
  `93fff15745bded35dd39e5d0802a38afefc5d8355f3b2d9892b59399839a0d49`:
  both end-to-end ownership tests and the new direct Rust entry failed as
  expected on the base version. The first smoke run mistakenly reused a Cargo
  target and failed its new-interface check; archive SHA-256 is
  `2e3152008807e6557acdc2c2c806d4502381bdc74235ce914e6ff2664ac33a18`.
  A fresh-target smoke then exposed an incorrect Python-list-to-Rust-tuple
  conversion, subsequently fixed; its failing archive SHA-256 is
  `7aa09f95686ea1ab5706614bb019bddc1f830e4e44d5e04672d3fb9e2352c7c9`.
  A further fresh-target smoke passed 29 tests; its archive SHA-256 is
  `3783895ebb65c57665f3288ce3e5754fbaa102398bd59deff6f283eb5c29bbe1`.
- The complete two-rank search passed Compute Sanitizer memcheck with zero
  reported errors per process and CUDA API reporting disabled
  (`--report-api-errors no`). This is not an unfiltered pass; the earlier
  unfiltered NCCL-initialization API error remains documented in
  `test_results/refactor_20261116/STAGE_CNOT_STREAM5_TRANSPORT.md`. The filtered
  memcheck archive SHA-256 is
  `5d218488f9bee374421be27a88ef1ac4b685f526573235741a92dcf27987a45a`.
- `cnot-native-goal-green/verify_results.py` validates the pinned sources,
  archived tests, rank receipts and provider lease-release receipt.
  Instance `60984269` was deleted and provider absence was verified. Its final
  reported duration and hourly rate imply approximately $0.044; final billing
  may differ.

This is an intermediate migration. User scorer calls, some tensor stages and
collectives, and old-library fallbacks still call Python/Torch. The evidence
covers one host with one/two RTX 3060 GPUs. Multi-node/Slurm and physical T4,
Hopper and Blackwell acceptance are not established by these tests.
