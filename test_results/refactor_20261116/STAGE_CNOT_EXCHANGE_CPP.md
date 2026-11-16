# CNOT exchange packet preparation in C++ ATen — 2026-11-16

The tested source is Git base `21057f447cd790a7d7b208a779f79a94d6d23b90`
plus five pinned overlays in `cnot-exchange-native/cnot-exchange-native-evidence.tar.gz`.
The remote job checked source bytes before and after execution. Archive SHA-256:
`80ea76c6a0c3c506f18033e957e49e34825a05e5c62bcf46120cef3d61e4a315`.
The adjacent verifier checks the bounded evidence and source overlays; pass the
published stage commit to check immutable source bytes.

The Rust CNOT controller uses C++ ATen for CUDA owner counts and stable grouping
of records before the existing rank exchange. The single-rank path, NCCL Stream
5 transport, process group and full-matrix identity remain unchanged. CPU and
older external operator libraries keep the existing path. This is a code
transfer, with no performance claim.

On Linux with PyTorch 2.9.0+cu128, CUDA 12.8 and one/two RTX 3060 12 GiB:

- Rust workspace, 34 focused CPU/Gloo tests, 54 CUDA cases and the complete
  Python integration suite (766 passed, 10 skipped) passed.
- Stable owner grouping matched the previous tensor operations for empty and
  singleton batches, repeated owners and unused destinations. A simulated
  older operator library exercised the fallback.
- Full one/two-GPU searches, two-rank NCCL radix and goal error/reuse tests,
  and checkpoint comparisons passed. The checkpoint Qhat case replayed
  `[5, 7, 0]`; proposal retained its feature F02 coverage rejection.

These results do not establish multi-node/Slurm behavior, other architectures,
all model/checkpoint domains, performance or completion of remaining Python
stages.
