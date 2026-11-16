# CNOT full-matrix distinct preparation in C++ ATen — 2026-11-16

The tested source is Git base `f6dd04684cb8e9fdcfcc775b240acc9cd28d29c8` plus
six pinned overlays in `cnot-distinct/cnot-distinct-evidence.tar.gz`. The job
checked 565 source files before and after execution. Archive SHA-256:
`f95f88b6d175112df6ae9392a7348dc3e67a33b9b54cdf914f5a1770a5d8c38d`.
The adjacent `verify_results.py` checks the evidence and source overlays; pass
the published stage commit to verify immutable source bytes.

The CUDA path now sorts received records by generation, constructs matrix
views and applies the existing full-matrix deduplication through a C++ ATen
stage. Child scoring and rank collectives remain Python callbacks. The CPU and
older external-operator paths remain available. There is no performance claim.

On Linux with PyTorch 2.9.0+cu128, CUDA 12.8 and two RTX 3060 12 GiB:

- Rust workspace, 34 focused CPU/Gloo tests, 26 CUDA cases and the complete
  Python integration suite (738 passed, 10 skipped) passed.
- CUDA comparisons covered empty inputs, repeated matrices, differences in
  later matrix words, noncontiguous closed sets and n=4/65/144. A simulated
  old operator interface exercised the fallback.
- Complete one/two-GPU searches, a two-rank NCCL radix comparison and the
  rank-failure/reuse test matched their references. The checkpoint scorer
  replayed `[5, 7, 0]` in Qhat and yielded the same proposal feature F02
  coverage rejection on one and two GPUs.
- The time guard stopped the first run during the full Python suite. The task
  lease was restarted, the interrupted output was retained on the instance,
  and the complete set was rerun against the pinned retry source.

Multi-node/Slurm, other architectures, all scorer domains, performance and
completion of all Python stages are not established by this evidence.
