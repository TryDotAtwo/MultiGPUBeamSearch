# Experimental session-owned NCCL identity check

Base checkout: `c6d2d92b1f6c60ef75f0103e28271495be4c4c0c` on `main`.
An earlier profiled repeated Rust-owned session spent 1.522 seconds in NCCL
discovery/identity for a 429,628,328-byte library; see
`test_results/refactor_20261107/STAGE5_PUBLIC_RUST_SESSION.md`. That profile
suggests a host bottleneck but is not a benchmark of this change.

The new option is explicitly enabled with
`NativeSession(experimental_parallel_nccl_identity=True)` for either session
owner. Its session-local entry contains the normal whole-file SHA-256 and a
SHA-256 digest per 1 MiB block. The initial check hashes both representations.
Every reuse rereads and hashes all blocks with up to four workers; the normal
digest is reused only when every block matches. On a mismatch the normal digest
and block digests are recalculated. Reuse relies on block SHA-256 collision
resistance. Header identity and the post-build standalone NCCL verification
retain their existing paths. An opted-in prepared runner also checks that its
recorded header and library remain available and match its build receipt.
File metadata is used to detect changes during a read, not to establish
equality between calls. As with ordinary file hashing,
undetectable concurrent mutations with indistinguishable filesystem metadata
are not a snapshot guarantee.

On Windows, initial tests before implementation failed as expected
(`nccl-red.log`). Review-driven tests also reproduced the prepared-runner and
session-ownership defects before their fixes (`nccl-pinned-red.log` and
`nccl-review-red.log`). Coverage includes empty/partial blocks, same-size edits
with mtime restored, replacement, size changes, detectable concurrent mutation,
alias retargeting, opt-in for both session owners, and the public dispatch to
prepared-runner and session reuse. Runtime and device fixtures in these tests
are mocked: they do not run CUDA searches. The final regression and any skipped
platform-specific tests are recorded in `nccl-final-01.log`: **152 passed,
3 skipped**. Two skips require POSIX process groups; the real symlink-retarget
test was skipped because this Windows account could not create file symlinks.
A deterministic path-resolution simulation passed, but is not a substitute
for the real POSIX symlink test.
Python's installed `pyreadline3` emitted a Windows console-handle exception at
interpreter shutdown after reporting the successful tests; pytest exited 0.

`nccl-cpu-diagnostic.json` records five alternating pairs on a synthetic 16 MiB
file, after the initial calls, and checks all returned hashes. This is a small
local CPU diagnostic, not a production library, cold-start comparison, GPU
result or full-step benchmark. It cannot justify enabling the option by default.

## Two-RTX-3060 evidence

The tested Git commit was `dbdd880817eb42730308330339348b5e4219881a`. On Linux,
the Python suite reported **570 passed, 9 skipped** (`vast-3060/cpu-tests.log`).
The public-session fixture passed for CUTLASS and LibTorch, both Python and Rust
session owners, and one and two physical RTX 3060 GPUs. It checked repeated
session use, an unsolved case, changed models and beam widths, collected paths,
and legal replay. The specialized inference implementation and precision were
unchanged. These bounded synthetic MLP checks do not cover the full CNOT,
Transformer, GNN or ensemble acceptance scope.
An independent local check replayed the stored paths and recomputed the paired
medians; its receipt is `vast-3060/result-verification.json`.

The paired measurement used a synthetic FP16 Hamming model, two GPUs, beam
width 4096, maximum depth six, one warmup and five alternating pairs of complete
public Rust-session calls. All measured cases were unsolved; an additional
solved case had equal legal paths. Median repeated-call wall time was 0.579843 s
without the option and 0.428234 s with it, a 26.15% decrease for this fixture.
The corresponding median worker times were 0.232345 s and 0.232646 s. The
measurement therefore supports a reduced host-side repeated-call overhead on
this host, not an inference-kernel speedup. The first calls were 2.029080 s and
1.947310 s, respectively, but used a shared existing build cache and always
ran in that order; they are not a controlled cold-build comparison. GPU memory
was sampled with **both** sessions resident: 636/12288 and 613/12288 MiB; this
is neither a peak nor a per-option comparison.

The benchmark does not isolate the GPU search step, count copies or allocations,
or establish behavior for other models and hardware. It is insufficient to
promote the option by default under the plan's performance gate. T4, Hopper,
Blackwell, Compute Sanitizer, and the remaining full-scope checks were not run
in this stage. The option remains explicitly experimental.

The first GPU attempt failed before launch because the remote sparse checkout
excluded the top-level CMake files. The next reached Rust session preparation but
failed because the restored prior-build artifacts lacked their CMake cache
files. Both failures and the successful retry are preserved in `vast-3060/`.
The missing files were restored from the pinned commit and the SHA-256-verified
prior archive, respectively. The native binaries and Rust worker were reused
from that earlier archive, whose bytes and current build identities were
checked; they were not rebuilt in this run. `vast-3060/source-verification.json`
checks the pinned source blobs and artifact hashes before and after the runs.

The full evidence archive on disk D has SHA-256
`e0ba5344fd25e36e02c7e8345f8033ccd48cb7ba83bef243a4f1942d43ccf7b3`.
Its 2,288 file records and 659 unique content blobs were verified, and sample
model and native-log files were restored and verified. Thirty-nine rebuildable
interpreter/NCCL snapshot files were omitted and identified in the manifest.
The compact logs, results and verification receipt are in `vast-3060/`. The
larger Rust/CUDA refactor remains incomplete.

Both task-owned Vast leases were destroyed and their absence verified through
fresh provider inventory; see `release-first.json` and `release-direct.json`.
Their elapsed-time charges are estimated at about $0.14 combined from the
provider durations and hourly rates. This excludes any separate traffic or
storage billing and is not an audited invoice or the total across earlier work.
