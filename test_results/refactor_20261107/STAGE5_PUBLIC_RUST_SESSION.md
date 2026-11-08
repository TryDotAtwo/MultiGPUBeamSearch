# Stage 5: opt-in Rust owner for public sessions

`NativeSession(engine="rust")` routes each whole resident search through the
Rust session and rank owner and the managed native C ABI library. The default
`NativeSession()` remains the existing Python owner. Model preparation,
profile selection and calibration remain in Python in both cases. The Rust
path requires the packaged Rust extension and worker and a managed native
build; a user-supplied external runner is not supported by this path.

The evidence below was obtained on an owned Vast instance with two physical
RTX 3060 GPUs using the checkout based on commit
`6921bfd07c26a593e9dec380cefdcb6b41adf005`. The tested bridge and GPU test
files are identified by SHA-256 in
`stage5-public-rust-3060/exact-inputs-{before,after}.sha256`. These files match
each other and the Git-normalized source bytes. An earlier complete run used
CRLF working-tree bytes for two Python files; after identifying the difference
the suite, GPU checks, package build and timing were repeated on the exact
bytes intended for Git.

* The Python integration suite passed 546 tests with nine skips. Fourteen
  focused session tests also passed after the test-only addition of injected
  rank-failure cleanup. These tests include mocked workers and are not GPU
  acceptance.
* For each of CUTLASS and LibTorch, the public session ran six deterministic
  cases on one and two GPUs with each session owner: 48 cases total. Cases
  covered a first search, compatible repeated search, unsolved input, model
  change, width change and collection mode. Both owners produced the same
  paths; collected path sets matched for all four backend/device combinations.
  Independent replay validated the returned paths. Worker reuse, release on
  configuration changes and cleanup on session close were checked.
* The checkout wheel and the wheel rebuilt from the source distribution both
  installed and contained the same 333-file native source manifest. Both
  Python package names, the Rust extension and worker, and construction of the
  experimental session were checked. The installed checkout wheel then ran a
  two-GPU CUTLASS first and repeated search with both owners and equal replayed
  paths. Its worker SHA-256 matched the worker used for the earlier GPU tests.
* A paired timing experiment used two simultaneously resident groups on the
  same two GPUs, a synthetic Hamming FP16 MLP, width 4096, six search depths,
  disabled autotuning and five alternating repeated calls. Median complete
  public-call wall times were 2.235718 s for the Python owner and 2.210314 s
  for the Rust owner, a difference of about 1.1%; this is not evidence of a
  material speedup. The recorded cold calls were 5.243044 s and 6.948157 s,
  respectively, but used a shared warmed build cache and fixed order, and are
  not a controlled cold-start comparison. With both groups resident, the
  recorded GPU usage was 599 MiB per GPU. Per-call data and run directories
  are in `stage5-public-rust-3060/paired-lf/`.

One additional Python cProfile measurement of a repeated Rust-owned call took
2.280 s in total; `prepare_runtime` accounted for 2.007 s, including 1.522 s
in NCCL discovery/identity, while the Rust session call accounted for 0.235 s.
This single profiled call identifies a likely host-side optimization target,
not a benchmark of a fix. The library identity and session invalidation
contracts must be preserved by any later reuse optimization. The profile is in
`stage5-public-rust-3060/profile-lf.log`.

The GPU fixtures use a synthetic model and bounded permutation problem; they
do not establish trained-model accuracy, broad workload performance, CNOT
semantics, multi-node behavior, cancellation on GPU, T4/Hopper/Blackwell
support, or readiness to switch the public default. Full-step optimization
promotion still requires the agreed separate paired performance gate.

The run also exposed a disk cost: a fresh worker group snapshots its verified
NCCL library into its private run directory. On this host each snapshot had
429,628,328 bytes and a distinct inode with one link. Compatible calls on an
already resident group do not create another such snapshot. Any replacement
must preserve source-mutation isolation and loader identity; no unverified
change to this behavior is included in this stage.

Compact evidence is in `stage5-public-rust-3060/`. The content-addressed raw
archive `.local/cache/vast-59637230/public-rust-exact-evidence.tar.gz` contains
6,024 records and 1,343 distinct blobs, including generated model fixtures,
runner/library artifacts and package distributions. All blob hashes were
verified on the host, and its archive SHA-256 is
`9db81b13f09ac174a9c4d557030f1391110b2a6487c9d8d8d4eaf751d606cd6e`.
The transferred archive matched this digest on disk D; a result JSON and a
synthetic weight were restored and matched their remote originals.
An earlier archive, `public-rust-evidence.tar.gz`, has SHA-256
`8eef67b3ad728a1041e37709be2266588e721432beda09804aae8a0d8a0fc073`;
a result JSON and a synthetic weight restored from it matched their originals.
Rebuildable CMake/interpreter files, installed-wheel copies and external NCCL
snapshots were excluded and listed in the archive manifest. Synthetic model
files and binary packages are retained in the local archive, not in Git.

An initial suite invocation and a GPU setup attempt failed because the sparse
checkout omitted tracked notebook and CMake inputs. Those failed logs are
preserved in the raw archive; the tracked inputs were restored from the base
commit before the passing runs. The failures were setup failures, not passing
GPU evidence.

The task-owned Vast instance 59637230 was destroyed after archiving, and a
fresh provider inventory confirmed its absence. The final pre-destruction
snapshot reported 4,016.16 seconds at $0.134222/hour, approximately $0.150
of time-based instance charges; this is not an audited final bill and does
not include possible ancillary charges. The sanitized provider receipt is
`stage5-public-rust-3060/release-receipt.json`.
