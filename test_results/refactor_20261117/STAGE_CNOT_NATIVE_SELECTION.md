# Rust-owned CNOT global selection stages

The Rust CNOT backend owns the score bits, keys, masks and batch handles used
by the existing global cutoff. The `beam-runtime` pass schedule, C++ ATen/CUDA
operations, stable ordering, score policy and collective sequence are unchanged.
PyTorch continues to own the tensor storage. The older Python stage adapter is
retained for older controller/extension combinations.

The green evidence uses base `f79469d3cca4374a5cd9b16bc2f739ec82bea620`
and eight SHA-256 source overlays, verified before and after execution. The
new Rust module was built into a fresh Cargo target and its direct selection
entry was checked before tests. The archive SHA-256 is
`9632ca233ac491485d086497549affeec04e7232c40f900d70a54acb286a82ff`.

- Rust workspace/build, 53 focused CPU/Gloo and CUDA-controller tests, and 62
  additional CUDA tests passed. The full Python suite reported 793 passed and
  10 skipped on Linux with CUDA 12.8, PyTorch 2.9.0 and two RTX 3060 GPUs.
- One- and two-GPU complete CNOT results matched the Python reference in five
  audited cases, including qhat, proposal and 144 wires. The legacy stage
  owner was used together by both ranks when one rank lacked the newer stage
  interface. An injected scorer error reached both ranks and the same group
  was reused afterwards.
- Direct selection comparisons cover signed zero, close and extreme finite
  scores, equal scores, all tested widths and empty owners. Three old CUDA
  operator combinations use the same result; a two-rank NCCL fixture also
  exercises different operator capabilities on the two ranks.
- The existing CNOT checkpoint-scorer fixtures, Stream 5 lifecycle and record
  exchange, radix and goal collective tests were run on one/two GPUs.
- The frozen pre-implementation red archive SHA-256 is
  `ec95aff5cb49cd7d4a3c773f4e02ea244d0f4aeab77657a67f6e48095769da3b`.
  It records the missing Rust selection owner and direct entry. Initial smoke
  SHA-256 is `202a8c426df0098d707d24ebe0b07ec13c86a53996fc6a6b70b8266b827fd1fe`;
  the subsequent legacy-operator smoke SHA-256 is
  `20c77b56a315d7f190f08cbccac7e951df53f6951cc5768b2cc5e342c1e7bb32`.
  The latter reused the prior smoke's Rust build; both have the same Rust
  source hashes. The final green run used a separate fresh target.
  The earlier full green run, before the mixed-operator tests were added,
  reported 790 passed and 10 skipped; its archive SHA-256 is
  `febbd981b89f19767ef8f7a5fcde3815d6021f27e8746698b90ec66e68db565c`.
- The complete two-rank search was also checked with Compute Sanitizer
  memcheck: zero reported errors per process, with CUDA API reporting disabled
  (`--report-api-errors no`). This is not an unfiltered pass; the earlier
  unfiltered NCCL-initialization API error is retained in
  `test_results/refactor_20261116/STAGE_CNOT_STREAM5_TRANSPORT.md`.
  The filtered memcheck archive SHA-256 is
  `4e01e7b6ccf432b55d7a08a2d6695b8c646608eeb604ef13f41d5716c53743dc`.
- `cnot-native-selection-green2/verify_results.py` validates the pinned
  sources, archived tests, rank receipts and provider lease release receipt.
  Instance `60980053` was deleted and its absence verified with the provider.
  The last reported duration and hourly rate imply approximately $0.072;
  final billing may differ.

This is an intermediate migration. Local-goal coordination, user scorer calls,
some collectives and old-library fallbacks still call Python/Torch. The evidence
covers a single host with one/two RTX 3060 GPUs; multi-node/Slurm and physical
T4, Hopper and Blackwell acceptance are not established here.
