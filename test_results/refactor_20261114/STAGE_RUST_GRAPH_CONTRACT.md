# Rust validation of permutation graph identity

`beam-runtime::SearchSession` checks that the declared graph hash is the
SHA-256 of the CayleyPy checkpoint representation of the permutation graph:
the center, ordered generators, and `PERMUTATION` type encoded with Python's
`json.dumps(sort_keys=True)` default separators. Start state and generator names
are deliberately absent. The implementation hashes this representation
incrementally without building another full graph JSON buffer.

The check runs before rank startup and before the trivial solved/exhausted
result. A stale but correctly shaped hash returns an error; a previously
resident rank group is released by the session's existing error path. This
changes the behavior of manually constructed Rust/PyO3 identities that supplied
a valid-looking unrelated hash. Valid hashes produced by the existing
`GraphContract.from_graph` retain their existing representation.

The check binds the **Rust identity description**, not the bytes subsequently
read by the native runner from its puzzle file. It does not cover CNOT, change
CUDA kernels or numerical precision, or make the default Python session use the
Rust owner. The previously documented MLP loading and cancellation limitations
remain. No performance promotion is asserted.

## Tests and source provenance

- The Rust and real-PyO3 RED logs capture acceptance of stale hashes before
  the change. Afterward, golden Python-generated hashes cover state sizes
  1, 2, 4, 11, 128, 129 and 150, labels including 255, repeated labels, and
  the maximum 255 generators. Mutated center, generator order/content and
  digest are rejected. Existing resident-process tests exercise cleanup on an
  invalid identity. Actual CayleyPy `GraphContract` objects with ordinary and
  repeated labels pass the Rust session; altered generators fail.
- On Windows, the two focused Rust integration binaries passed five tests;
  the real PyO3 session test passed; the graph-contract and existing contract
  Python tests passed 23 tests using the local `CayleyPy` checkout. The first
  local collection used an installed CayleyPy missing `models.checkpoint` and
  failed before tests; that log is retained separately. These are scoped CPU
  checks, not a Windows full-workspace result.
- On Linux/CUDA 12.8, `cargo test --workspace --locked` passed with one ignored
  CPU ABI fixture; release PyO3 and worker builds passed; PyO3 session,
  cancellation and SIGINT cleanup passed; the complete integration suite
  reported **643 passed, 9 skipped**. The GPU test used an FP16 synthetic
  Hamming MLP, logical beam 64, depth limit 4 and the CUTLASS backend. Two
  public calls each on one and two RTX 3060s solved their inputs, independently
  replayed their paths, and reused the same per-session PIDs. Rank bootstrap
  receipts confirm rank count and effective width. This is a bounded public
  correctness test, not a saturated-step benchmark, Compute Sanitizer run,
  trained-model quality test, or hardware acceptance for T4, Hopper or
  Blackwell.
- The tested base is Git commit
  `db346e55cf06f567f059fc74f4ef6b7986ededc9`; the source package contains 494
  files from that commit and seven source/test overlays. SHA-256 checks were
  performed after extraction and after the run. The independent
  `rust-graph-contract/verify_results.py` checks the base files against Git
  blobs, verifies the overlays and selected archived evidence, associates all
  three bootstrap logs with their corresponding public sessions, and replays
  all four paths. Reproduction requires Git, the pinned base commit, and the
  local source archives in `.local/cache/rust-graph-contract-20261115/`.
- The complete verified evidence archive on D is
  `.local/cache/rust-graph-contract-20261115/evidence.tar.gz`, SHA-256
  `613c1ec5fc1a37f9ec3c3969cace19795bb08354b49cf7db5b1a410da76815fa`.
  Its 97 distinct content blobs and 163 records were checked; 938 rebuildable
  intermediate files are listed as omitted. The portable bootstrap receipt has
  SHA-256 `4cff0ad67a4c0ea49a31a36dc26822a0ef3bcae67ca88705f8433b547f0dc705`.

Vast lease `60702743` used two RTX 3060s at $0.1688889/hour and was destroyed
after archive verification; fresh provider inventory confirmed its absence.
The last recorded running duration was 781.27 seconds and verified absence
followed within 866.3 seconds from the provider start timestamp. Those bounds
correspond to approximately **$0.0367–$0.0406** for GPU/disk, excluding
network and without asserting an invoice. A local 90-minute stop guard was
started before rental but was no longer present at final cleanup, so its
continued operation was not verified.
