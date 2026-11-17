# Standalone Rust-owned CNOT CUDA domain

The optional C ABI and `beam-native-sys` owner encapsulate one thread-bound
CUDA device, stream, fixed-capacity state/descriptor/result buffers and queried
CUB scratch. They call the existing CNOT generation, owner hashing and
full-word distinct kernels. Inputs and outputs in this slice are synchronous
host slices, and the public CNOT search still uses its existing GPU-resident
Torch stage backend and multi-rank collective protocol.

The source-pinned main run uses base `ac9a571388a0440d406fc6a05cefdd48821b2531`
plus overlays. The standalone final test adds an overflow case and optional
skip check while using the identical tested shared library. Evidence archives:

- Main: `8ffc4825d5dcb69d2054a7f4e853c3980be51396ccbbe2659d7b8fbb6fc12a6d`.
- Final domain test: `d21656ea740676d5f02e6f5f9e8dbdf7070f2936a4bd92a23baead8461dfffea`.
- `cnot-owned-domain5/verify_results.py` verifies the tested source, archives,
  sanitizer summaries, search receipts and provider release.

On one host with two RTX 3060 12 GiB GPUs, CUDA 12.8 and driver 580.178.04:

- Domain tests passed for n=2, 4, 64, 144 and 256, including partial tail
  words, original child IDs, full-matrix equality, independent device owners,
  zero capacities, overflow, wrong-thread rejection and current-device
  restoration. These are domain checks, not a new complete CNOT search.
- Compute Sanitizer memcheck reported 0 errors and 0 leaked bytes; racecheck
  reported 0 hazards in this test workload. The Rust workspace and the explicit
  GPU ownership/reuse test passed, as did formatting.
- Existing CNOT searches matched their reference on one/two GPUs; 55 focused
  Python tests passed. The full integration suite reported 833 passed and
  12 skipped. The native checkpoint-scorer fixture was not built for this
  run; this result does not replace its prior separate acceptance evidence.
- The standalone library compiled for SM75, SM80, SM90 and SM120. Those are
  compilation results only: physical T4, Hopper and Blackwell execution,
  multi-node/Slurm and CNOT search using the new standalone owner remain open.

The owned Vast instance `60993089` was destroyed; provider absence was verified.
Its last reported duration and rate imply approximately $0.064 of instance
charges, excluding potential fees and traffic. Earlier setup attempts and
their failures are retained under ignored `test_results/` directories.
