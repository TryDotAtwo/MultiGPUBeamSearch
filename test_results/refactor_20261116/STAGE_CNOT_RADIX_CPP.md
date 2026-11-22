# CNOT radix tensor stages in C++ ATen — 2026-11-16

The tested source is Git base `2acca50be94c1664325f794d653339c9393e43fc`
plus four pinned overlays in `cnot-radix-native/cnot-radix-native-evidence.tar.gz`.
The remote job checked source bytes before and after execution. Archive SHA-256:
`4ab8251c503ec5ba63d13a3394762381c28af399e3aad3313c18b8901b206461`.
The adjacent verifier checks the bounded evidence and source overlays; pass the
published stage commit to check immutable source bytes.

The Rust CNOT controller still controls radix passes and the same collective
operations. CUDA histogram and mask-update tensor sequences are now implemented
in C++ ATen using the same operations. CPU, the Python reference, and older
external operator libraries keep the existing paths. This is a code transfer,
with no performance claim.

On Linux with PyTorch 2.9.0+cu128, CUDA 12.8 and one/two RTX 3060 12 GiB:

- Rust workspace, 34 focused CPU/Gloo tests, 49 CUDA cases and the complete
  Python integration suite (761 passed, 10 skipped) passed.
- CUDA comparisons covered radix shifts, tied and negative keys, mask updates,
  partial and empty cases and a simulated old-operator fallback. Two-rank NCCL
  radix selection passed three comparison cases on both ranks.
- Full one/two-GPU searches, native and reference two-rank goal error/reuse
  tests, and checkpoint comparisons passed. The checkpoint Qhat case replayed
  `[5, 7, 0]`; proposal retained its feature F02 coverage rejection.

These results do not establish multi-node/Slurm behavior, other architectures,
all model/checkpoint domains, performance or completion of remaining Python
stages.
