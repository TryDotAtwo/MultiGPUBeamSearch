# Opt-in Rust validation for CNOT inputs and construction paths

`search_cnot(..., validation="rust")` and the corresponding CLI option use an
owned `beam-core` packed matrix to validate input and independently replay a
returned construction. Geometry is 2–256 wires, with little-endian 64-bit row
words, zero tail bits, and a nonsingular GF(2) matrix. Physical action IDs use
the existing control-major alphabet; replay starts at identity and compares all
matrix words. A hash is not used to decide matrix equality. PyO3 borrows the
Python byte buffer only during construction; the Rust value retains its own
words. Elimination has a separate temporary copy, at most 8192 bytes. The host
byte snapshot is still retained by the current Python search for its lifetime.

All ranks must select the same validator. An unavailable/older Rust extension
fails rather than silently selecting Python. The default remains the original
Python validation. The Python search loop, PyTorch collectives, specialized
CNOT CUDA operators, and the separately supplied trained checkpoint scorer are
unchanged. This is an incremental control migration, not a native CNOT engine.
The snapshot does not bind the subsequently used GPU tensor against a concurrent
external mutation; no new guarantee is claimed for that pre-existing boundary.

## Verification

- The Rust workspace tests and release PyO3/worker builds passed on Linux;
  two native-ABI fixture tests were ignored in this run (they passed in the
  preceding native-graph-binding stage).
  The real extension matched an independent Python integer-row replay for 88
  seeded paths across eleven geometries, validated all 512 binary 3x3 matrices
  against independent elimination (168 invertible), rejected padding, truncated
  or extended payloads, singular matrices and invalid action IDs, and survived
  release of the borrowed Python bytes.
- The full Python integration suite reported **643 passed, 11 skipped**. The
  same source-bound run compared the entire returned result for Python and Rust
  validators on one CPU and two CPU/Gloo ranks, plus one RTX 3060 and two
  RTX 3060/NCCL ranks.
  The fixtures included both heads, full audit output, a 144-wire two-gate path
  with a cross-rank history route, and a 144-wire single-gate completion. All
  twelve solved rank paths were independently replayed. Mismatched validators
  and a singular input on rank 1 failed collectively in both two-rank runs.
- The scorer in these searches was the synthetic `WireFixture`; no trained CNOT
  checkpoint, other GPU family, multi-node machine, sanitizer run, or sustained
  beam performance was evaluated. No acceleration or default promotion is
  claimed. Local small CPU tests additionally cover the CNOT alphabet boundaries
  and error cases. The first remote run passed but lacked a returned multi-gate
  path, so it was retained separately and repeated with the stronger fixture.

## Source and resource record

The tested base is commit `7ae022d9383e31d1ee673f567c39443ad8433b3c` plus ten
source/test overlays. `cnot-rust-validation/verify_results.py` checks the full
portable archive against its SHA-256, its members and any extracted copies, the 502 base Git
files, all ten overlays, CPU/GPU rank records and independent path replay.
The final archive SHA-256 is
`d767e36f9238ac8d8b8d8715821f5eb1780e050e6176a4c694604c253a64f456`;
the initial archive is retained on D with SHA-256
`d7438995d9a40527c9f9830648199d61fabe7856250bb6031239077b957dee59`.

Vast lease `60715196` used two RTX 3060s at $0.151111/hour. The complete
archives were downloaded and verified before the instance was destroyed;
fresh provider inventory confirmed absence. GPU/disk cost for the recorded
interval is approximately **$0.0517–$0.0528**, excluding network and without
claiming invoice totals. The task-specific timeout guard was stopped after
provider absence was confirmed.
