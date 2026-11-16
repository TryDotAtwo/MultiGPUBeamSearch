# CNOT selection preparation and ordering in C++ ATen — 2026-11-16

The tested source is Git base `0e8af0c762c2930e4c4989d3a667c848b02feb9a`
plus four pinned overlays in `cnot-selection-stages/cnot-selection-stages-evidence.tar.gz`.
The remote job checked source bytes before and after execution. Archive SHA-256:
`73f1d328a5b1feefb647177d8438dfca1c278ea842b13b1b67b81ecb2e5938d9`.
The adjacent verifier checks the bounded evidence and source overlays; pass the
published stage commit to check immutable source bytes.

The Rust CNOT controller uses C++ ATen for CUDA selection key preparation,
selected-record construction and final ordering. The same numeric conversion,
signed-zero handling, stable sort sequence, rank collectives and cutoff logic
are retained. CPU and older external operator libraries keep the existing path.
This is a code transfer, with no performance claim.

On Linux with PyTorch 2.9.0+cu128, CUDA 12.8 and one/two RTX 3060 12 GiB:

- Rust workspace, 34 focused CPU/Gloo tests, 62 CUDA cases and the complete
  Python integration suite (774 passed, 10 skipped) passed.
- Selection comparisons cover empty and singleton batches, signed zeros,
  equal scores with distinct words, narrow and oversized widths, extreme
  finite values and a simulated older operator library. Tests assert that
  the new operators are used when available.
- Full one/two-GPU searches, two-rank NCCL radix and goal error/reuse tests,
  and checkpoint comparisons passed. The checkpoint Qhat case replayed
  `[5, 7, 0]`; proposal retained its feature F02 coverage rejection.

These results do not establish multi-node/Slurm behavior, other architectures,
all model/checkpoint domains, performance or completion of remaining Python
stages.
