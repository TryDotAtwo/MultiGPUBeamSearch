# Rust-controlled CNOT Stream 5 record exchange

The Rust CNOT controller now selects a thread-bound Rust-owned communicator
after rank-wide capability agreement and controls the existing NCCL count and
payload exchange. If any rank lacks the compatible exchange ABI, all ranks
retain the legacy transport. The Python reference, CPU and single-rank paths
remain available. This changes ownership and control; it does not change the
CUDA kernels, precision policy, stable routing, matrix equality or NCCL protocol.

PyTorch owns the CUDA storage. `beam-native-sys` checks tensor layout, device,
capacity and stream and records the stream before handing borrowed addresses to
the native ABI. The Rust search and exchange controllers do not receive device
addresses. Two rank-wide votes precede count and payload transfer, respectively;
bad counts or 32-bit offset overflow fail before sending payloads. The existing
native owner remains bound to its creating thread and restores the caller's
device after close or abort.

## Evidence

- The tested base is `b68d37ce3e539cfc50a74167d7cc7e33054c228c`, with 15
  source overlays SHA-256 pinned in the evidence manifest. All source files
  checked before and after the run match. The archive and its verifier are in
  `cnot-transport-pass/`.
- Green evidence SHA-256:
  `13669543b557fde0f18002836f87dc8431d1e07df148a7e3618b6f4670baa46a`.
- On Linux with CUDA 12.8, PyTorch 2.9.0, NCCL and two physical RTX 3060s,
  the Rust workspace and extension, 43 focused CPU/Gloo tests and 62 CUDA
  tests passed. The full Python suite reported 783 passed and 10 skipped.
- Both ranks compared five complete CNOT searches against the Python reference,
  exercised the Rust transport, and agreed on legacy transport when one rank's
  Rust capability was disabled. The same rank group was reused after an
  injected scorer failure. A direct two-rank transport fixture covered unequal
  and zero peer counts, an empty exchange, a delayed rank, collectively rejected
  bad counts, and a successful exchange after the rejection.
- The existing two-rank communicator lifecycle, selection, native and legacy
  goal collectives, and the one- and two-GPU checkpoint-scorer comparisons are
  included in the same archive.
- The pre-implementation red archive is
  `cnot-transport-red/artifacts/cnot-transport-red-evidence.tar.gz`, SHA-256
  `cf065125ff90e37bb322992bc1161ae180bc76215edc33c62b4fd89ecdce4a1f`.
  It records the missing Rust exchange API and missing exchange controller.
- The first attempt failed on a Python `bytes` conversion, fixed by passing
  integer values. Its archive under `cnot-transport-green/initial-failure/`
  has SHA-256 `c9cb3d50cf8cb534ac28d0f61e8fedfd37f6e36aabca154380c4ac3d38e8b713`.
  The next attempt exposed a test assumption that a valid result always has
  gates; its archive under `cnot-transport-final/` has SHA-256
  `a0687c98fc835ceb2f67d7cab01310a2c9ff12a77a1594f6f1a76673a6cd497c`.
  The corrected test compares the full result including unsolved cases.
- The unfiltered Compute Sanitizer memcheck completed the exchange on both
  ranks but failed its error-exit gate. Both logs report CUDA API error 209
  (`cudaErrorNoKernelImageForDevice`) at `cudaFuncGetAttributes` in NCCL
  initialization; the archived report has SHA-256
  `0e3d5aac8d065e254849138c604b6b3aebe7f2e147c91fe7218cfcd323f5995d`.
  A second memcheck with `--report-api-errors no` completed the same exchange
  with zero reported errors on both ranks. Its archive SHA-256 is
  `d8757cdb74a9c34a68d056d1865a5fe6137d4fbda8d8908c4349202a3d02fcc1`.
  Because CUDA API errors were suppressed, this does not establish an
  unfiltered sanitizer pass.
- Vast instance `60969024` ran these checks; its provider release receipt is
  `cnot-transport-pass/release.json`. Its reported running duration and
  $0.16296/hour rate correspond to approximately $0.151; final billing may
  differ. Provider absence was verified after deletion.

This is an intermediate migration stage. Some tensor operations, scorer calls
and collective management still use Python callbacks; failure of the NCCL
communication itself and recovery of a failed GPU are not covered. A failed
native close can consume its handle without releasing the underlying resource.
Multi-node/Slurm and GPU families other than RTX 3060 are not validated here.
