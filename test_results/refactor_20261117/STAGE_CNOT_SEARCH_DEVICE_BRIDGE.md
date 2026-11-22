# Opt-in CNOT search device selection bridge

The Rust CNOT search can explicitly load a compatible CNOT domain library from
`MGBS_CNOT_OWNED_LIBRARY` on each CUDA rank. Its first-unseen stage borrows
rank-local Torch tensors and the current stream, uses the existing full-matrix
CUDA sort/compact implementation, and synchronizes before returning. The
adapter reads the device count on the host to form the selected tensor view.
The default path, CPU execution, Torch-owned tensors, NCCL transport and
checkpoint scorer remain in place. This is a migration step, not a completed
Rust/CUDA rewrite of CNOT or the permutation search.

On two RTX 3060 12 GiB GPUs, CUDA 12.8, PyTorch 2.9 and Rust 1.100, the
source-pinned runs checked:

- One- and two-rank audited search comparisons against the existing path for
  4 and 144 wires, qhat and proposal modes, repeated searches and mixed ranks.
  They checked result equality and legal solution replay where applicable.
- The existing two-rank CNOT equivalence suite and the native checkpoint
  scorer fixture. An invalid library on rank zero propagated a coordinated
  error to both ranks, and the search was reusable afterwards.
- The Rust workspace tests and formatting check. Standalone borrowed-buffer
  boundaries and the full CNOT identity cases are documented separately in
  `STAGE_CNOT_BORROWED_UNSEEN.md`.

The first attempted build was stale because overlay timestamps did not force a
Cargo rebuild. The subsequent forced build exposed forbidden `unsafe` blocks
in the Python crate; they were moved to the FFI crate. Both failed attempts
and the corrected rebuild, including the binary marker, are retained.

Default Compute Sanitizer memcheck reported one `CUDA_ERROR_INVALID_HANDLE`
from `cuKernelGetFunction` in the existing Torch CNOT owner operator. An
isolated call to that operator, without the new bridge, reproduced the same
error and returned its expected output. The bridge search completed under
memcheck with `--report-api-errors no` and zero reported memory errors; this
does not establish that CUDA API-error reporting is clean. The original
diagnostics and the controlled rerun are retained. Multi-node, Slurm, and
other physical GPU architectures were not verified by this integration run.

The source manifests, exact command logs, rank receipts, archive hashes and
failed runs are in `cnot-unseen-bridge/`; `verify_results.py` checks them
against the pinned base Git tree and final source overlays. The task-owned
Vast instance was destroyed and its absence verified in
`cnot-borrowed-unseen/release.json`.
