# CNOT batch expansion in C++ ATen — 2026-11-16

The tested source is Git base `93a3077da21fb48cc7d52089d08b9f7354e99134`
and six pinned overlays in `cnot-expand/cnot-expand-evidence.tar.gz`. The remote
job checked 563 source files before and after execution. The archive SHA-256 is
`720fb7c49c6af10556f422c682cce41323339f07d53229273d69830024039394`.
The adjacent `verify_results.py` checks the evidence and source overlays; pass
the published stage commit as its optional argument for immutable sources.

The Rust controller now calls a batched C++ ATen expansion stage on CUDA. It
uses the existing children and owner kernels, the same stable ordering and
physical CNOT routes and the original full-matrix representations. Python still
calls the batch scorer and supplies collectives. CPU uses the reference stage;
an older operator library without the new symbol is handled by the existing
implementation. This is an architectural transfer, not a performance claim.

On Linux with PyTorch 2.9.0+cu128, CUDA 12.8, NCCL and two RTX 3060 12 GiB:

- The Rust workspace, 34 focused CPU/Gloo tests, 19 CUDA packet/expansion
  cases and the full Python integration suite (731 passed, 10 skipped) passed.
- Expansion records and owners matched the reference for empty and nonempty
  inputs, n=4/65/144, full and partial alphabets, tied scores and both modes.
  A simulated old library interface exercised the fallback; an actual old
  external binary was not tested.
- One- and two-GPU complete searches and two-rank radix tests matched their
  references. The two-rank failure/reuse test passed. With the existing
  checkpoint scorer, Qhat replayed `[5, 7, 0]` on one and two GPUs and proposal
  produced the same feature F02 coverage rejection as the reference.

These checks do not establish multi-node/Slurm behavior, other GPU architectures,
all scorer domains, performance or completion of the remaining Python stages.
