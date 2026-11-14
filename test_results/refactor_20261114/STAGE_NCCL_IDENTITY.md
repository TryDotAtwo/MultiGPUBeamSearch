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

Vast and GitHub were intermittently unreachable during this stage. Until real
hardware evidence is added, target-hardware tests, unchanged-result comparisons,
resource measurements and five-pair full-call benchmarks remain required before
any performance claim or promotion. The larger Rust/CUDA refactor remains
incomplete.
