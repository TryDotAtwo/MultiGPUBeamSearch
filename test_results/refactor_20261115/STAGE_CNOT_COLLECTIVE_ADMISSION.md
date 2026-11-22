# Collective CNOT admission failures

Previously, each rank validated its matrix through `World.checked`, then read
the scorer's metadata and serialized its settings outside that boundary. If a
metadata property raised or a fingerprint value could not be encoded as JSON on
only one rank, that rank exited while its peers entered the digest all-gather.
The process group then failed or waited for its timeout.

Input validation, metadata reads and fingerprint construction now happen inside
the same existing rank-coordinated admission callback. Successful fingerprint
bytes are unchanged, and no additional collective was added. A single-rank
call retains the original exception behavior. Full-matrix CNOT equality,
search, inference and CUDA operators were not changed.

## Verification

- Before the change, two focused boundary checks failed locally and all four
  boundary/Gloo tests failed on Linux. The Gloo tests injected a failing
  metadata property on rank 0 and a non-JSON fingerprint on rank 1; the ranks
  did not both receive the intended coordinated error. The fixed boundary
  checks passed locally, and the Linux integration suite reported **674 passed,
  11 skipped**.
- The actual PyO3 validator and public CNOT search were exercised on one CPU,
  two CPU/Gloo processes, one RTX 3060, and two RTX 3060/NCCL processes. Both
  metadata faults were injected for each of the Python and Rust validators.
  Every participating rank received the rank-specific error and then completed
  a new search with the same process group. Validator disagreement and a
  singular rank-1 matrix were still rejected collectively.
- Python- and Rust-validated results agreed for the synthetic `WireFixture`
  cases, including a 144-wire cross-rank route. The independent receipt
  verifier replayed 12 returned paths against all original matrix bits and
  checked 16 rank-error receipts. The tests did not exercise a trained CNOT
  checkpoint, multi-node execution, other GPU families, Compute Sanitizer or
  search throughput. They do not establish a speedup or completion of the
  Rust/C++ CNOT search migration.

## Provenance and resource use

The tested source is base commit `cfba6e472093735dad0acfcb6fcf26bbb135be6b`
plus the three overlays in the evidence manifest. The downloaded archive is
`cnot-admission/remote-evidence.tar.gz` with SHA-256
`2cb231152d4b251dc74414fd2fc725329de4018b5a16e89442605f05b4ca9c41`.
`cnot-admission/verify_results.py` checks all 538 archived base files against
Git, the three current overlays, red/green status, rank receipts and independent
construction replay. It prints a short verification result.

The cheapest advertised two-RTX-3060 lease `60747812` did not provide a usable
SSH connection and was deleted without running the tests. The next offer,
`60748392`, ran the tests at $0.167259/hour. Both were destroyed and fresh
provider inventory confirmed absence. The provider reported **$0.046** for the
two task leases at query time; later invoice adjustments are possible. The
filtered current total for 25 known leases for this refactor was **$4.035**,
which is not a guarantee that every project charge has been identified or that
the invoice is final.
