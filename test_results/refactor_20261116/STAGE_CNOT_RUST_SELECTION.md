# CNOT Rust global radix selection — 2026-11-16

The tested source is Git base `4a2abfabb080f9d8121ec1cfb9c509c2268a0c12`
and the seven pinned overlays in `cnot-rust-selection/cnot-rust-evidence.tar.gz`.
The remote job checked 560 source files before and after execution. The archive
SHA-256 is `7dea593d68b820b084eb607b51328163d26fe394928bc273d6a4a3728ccd7a38`.
The adjacent `verify_results.py` checks the evidence and source overlays. Pass
the published stage commit as its optional argument to verify immutable sources.

Rust controls the existing global selection pass sequence, remaining width and
cutoff decisions. Torch computes each batched histogram and Python performs the
same rank collectives; exact keys, score precision, ties and CNOT matrix equality
remain unchanged. Older PyO3 controllers and older Python stage adapters keep
their respective existing selection path. This stage does not complete the
Rust/C++ migration, and does not establish a performance improvement.

On Linux with PyTorch 2.9.0+cu128, CUDA 12.8, NCCL and two RTX 3060 12 GiB:

- The Rust workspace passed; 34 focused CPU/Gloo tests, eight CUDA frontier
  cases and the full Python suite (720 passed, 10 skipped) passed.
- A separate two-rank NCCL comparison matched the original selection for all
  widths in three cases, covering an empty rank, equal scores, signed zero and
  finite scores near floating-point extremes.
- One- and two-GPU full searches matched their Python references, including
  Qhat, proposal, full alphabet and a 144-wire case. A failure on one rank
  propagated and the process group was subsequently reused.
- The existing checkpoint scorer matched on one and two GPUs: Qhat replayed
  `[5, 7, 0]`; proposal failed with the same feature F02 coverage rejection on
  both controllers. The checkpoint SHA-256 is
  `92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4`.

Multi-node/Slurm, other GPU architectures, all scorer domains and cancellation
during a collective were not established by these checks. Hardware costs and
cleanup for the shared lease are recorded in `STAGE_CNOT_DISTINCT.md`.
