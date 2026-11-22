# Bind Rust sessions to the native-loaded permutation graph

The permutation `SearchSession` first validates its CayleyPy graph SHA-256 and
then passes that digest to each rank. The C++ context copies the 64 lowercase
hexadecimal bytes through an optional ABI-1 extension before preparation. The
native rank computes the same canonical digest from its already loaded center
and ordered generators and refuses preparation if the values differ. Hashing
streams through a bounded buffer; it makes neither a second full graph JSON
buffer nor a Rust-to-C++ graph copy. Logical state bytes participate; storage
padding and generator names do not.

A Rust session requests a worker capability marker and replaces any caller
environment value for the expected graph hash. An older worker or a native
library missing the extension fails closed for this session. The worker's
ordinary `--resident-protocol-info` response remains unchanged. An arbitrary
external worker can claim a capability without implementing it; this contract
therefore applies to the matching Rust worker and native library used by the
managed public launch, not to untrusted custom executables.

The bound Rust search identity targets its center. A different target override
and the separate repair mode are rejected when graph binding is active; the
existing non-bound native CLI and its repair mode retain their previous
behavior. A concurrent in-place file write is not made atomic, but the graph
that was actually parsed and loaded is checked. CNOT, inference kernels,
numerical precision and the default Python session are unchanged.

## Checks and scope

- Python-generated canonical hash goldens for 120 positions / 18 moves and
  150 positions / 255 moves passed on Windows and Linux; the 96 / 24 build
  passed through CTest. Tests cover changed center and generator, padding
  exclusion and truncated storage. Linux ASan/UBSan passed the 120 / 18 host
  case. The real ABI fixture checked borrowed-hash copying, invalid digests,
  phase restrictions, and an older library with no extension.
- Linux `cargo test --workspace --locked`, the two explicit native-FFI fixture
  tests, PyO3 cancellation and SIGINT checks passed; the full Python suite
  reported **643 passed, 9 skipped**. An initial detached test run failed its
  SIGINT check because that harness inherited `SIG_IGN`. A direct rerun passed,
  and resetting SIGINT before the test interpreter fixed the harness. The
  original failed log, signal comparison and verified archive are retained.
- Four bounded public CUTLASS/FP16 synthetic Hamming searches, two each on one
  and two RTX 3060s, produced independently replayed paths and reused their
  rank PIDs on repeated calls. Five negative cases rejected a changed
  generator, center, target, repair mode, and changed center on only rank 1 of
  a two-rank group. The supervisor reaped all tested rank PIDs.
- An unfiltered NVIDIA Compute Sanitizer **memcheck** run covered one bounded
  public one-GPU search. The detailed report identifies the actual Rust
  `beam-runner` PID, reports zero errors and zero bytes leaked, and the returned
  path was replayed. This is not racecheck, multi-GPU sanitizer coverage, or a
  general proof of memory safety. The initial shorter sanitizer report did not
  identify the target PID; the detailed rerun supplies that evidence.

These are synthetic correctness checks, not measurements of full-step speed,
trained-model quality, multi-node behavior, CNOT semantics, or hardware
acceptance for T4, Hopper or Blackwell. No GPU optimization or speedup is
claimed.

## Provenance and cost

The tested base is commit `d98b4900a8ba3c7babb300700379ff6b0cd5eb80`, with 500
base files and 18 source/test overlays. Both were checked against SHA-256 before
and after the remote run. `native-graph-binding/verify_results.py` checks the
base against Git, the overlays, the archived receipts, all four paths and all
five negative cases. `verify_sanitizer.py` checks the actual attached PID,
sanitizer output, replay, and the recorded binaries against the same archive.

The complete verified archive is
`.local/cache/native-graph-binding-20261115/evidence.tar.gz`, SHA-256
`bd6dbfb4e54fb971b722ae89e1ccdc65694fb964b8fcc03b18627a51ce99ea32`;
its 125 content blobs and 221 records were checked, with 955 explicitly listed
rebuildable intermediates omitted. The portable trace archive has SHA-256
`86454356753668fd17b8517487ba8bfba430e18b968e1c8f8bb5858bfc27a598`.
The detailed sanitizer archive is included next to the verifier and has SHA-256
`386c2249e3480f5f889484537193a9352d64e799ba3f953b47eb7afd3eda8604`.
The first, SIGINT-inherited run was preserved as
`.local/cache/native-graph-binding-20261115/evidence-sigint-red.tar.gz`,
SHA-256 `30fbcb032fd5c203098bd487ec184308363bfe958395a3db245f5d7b1f1c3356`.

Vast lease `60710211` used two RTX 3060s at $0.1368889/hour and was destroyed
after archive verification; fresh provider inventory confirmed absence. The
last duration and verified-absence time bound GPU/disk cost to approximately
**$0.0743–$0.0755**, excluding network and without asserting an invoice. The
task-specific stop guard stayed live until the instance was confirmed absent.
