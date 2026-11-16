# Device-bound CNOT Stream 5 ownership

The experimental Rust communicator owner now uses a separate versioned C ABI
whose native handle records the CUDA device used for NCCL initialization. On
close or abort it temporarily selects that device and then restores the
caller's current device. Rust remains bound to its creating thread and keeps
the library loaded until the communicator is consumed. The raw ABI used by
the existing Python CNOT transport is unchanged. The experimental owner still
does not exchange GPU buffers or drive production search.

## Evidence

- Base commit: `9369f474683a518d019c4c55b3aebcfee0b0282b`. Four tested source
  overlays are SHA-256 pinned in the evidence manifest. The complete source
  checks before and after the run match.
- Green evidence: `cnot-device-green/cnot-device-green-evidence.tar.gz`,
  SHA-256 `00178cbf69d74973a450d2b3834165a7164d4c7d36755b573e4a87457a2c6c69`.
  `cnot-device-green/verify_results.py [commit]` verifies the archive, source
  overlays, red test, rank receipts and provider-release receipts.
- On Linux with CUDA 12.8, PyTorch 2.9.0 and two physical RTX 3060s, the Rust
  workspace and PyO3 extension built, 42 focused CPU/Gloo and 62 CUDA tests
  passed, and the full Python suite reported 782 passed and 10 skipped.
- Both NCCL ranks completed three lifecycle rounds with the calling thread's
  current device changed after initialization: explicit close, explicit abort
  and abort on drop. The test checked that the changed caller device was
  restored, and the existing Torch communicator still completed an all-reduce.
  Mock ABI cases cover the separate owned ABI, missing/incompatible versions,
  identity/rank validation and injected native errors.
- Existing one/two-GPU CNOT searches, rank collectives and the checkpoint
  scorer also passed. Their exchange path remains the existing Python-managed
  transport and does not validate Rust-owned payload buffers.
- The red test on the pinned base failed because the previous Rust owner did
  not call the device-bound ABI. Its archived evidence has SHA-256
  `856b5800e4628b76fbf4ce56956aac5a49a2567d5dcff535cff65977809524c0`.
- Vast instance `60965903` went offline during image setup and was removed
  without running tests. Replacement `60967063` ran the checks and was also
  removed. Both provider absence checks are recorded next to the verifier.
  The reported pre-deletion running durations at $0.1615/hour correspond to
  about $0.046 combined; final provider billing may differ.

Failed close consumes the native handle; if CUDA device selection or NCCL
cleanup fails, the underlying resource may remain allocated and this version
cannot retry it. GPU payload ownership, coordinated transport switching,
multi-node/Slurm and other GPU families are not validated by this stage.
