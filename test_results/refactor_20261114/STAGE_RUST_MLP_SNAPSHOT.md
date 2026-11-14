# Experimental Rust-owned MLP snapshot

For the existing default-OFF snapshot option, an explicit MLP `NativeModel` in
strict native mode now uses `beam-runtime` for snapshot identity, bounded copy,
staging, reuse and cleanup when the matching Rust extension is present. Python
continues to validate the manifest, tensor sizes and finite values, and the
retained snapshot is checked again immediately before submitting a search.
An older extension without `MlpSnapshotStore` keeps the Python implementation.
The CUDA inference policy and specialized kernels did not change.

The Rust store hashes the manifest, graph digest, lexically ordered tensor names
and tensor bytes while copying in 64 KiB chunks. It reopens and checks the
staged bytes before publishing. Source equal to the target is checked and
reused without a copy. A non-existent cache identity does not need to be
created. On Unix files are opened nonblocking and the opened object is checked
to be a regular file. Linux publication uses atomic `RENAME_NOREPLACE` so a
concurrently created target is not replaced; a filesystem or kernel without
support returns an error. Outside Linux the pre-existing platform rename
semantics remain; Slurm network filesystems have not been checked.

Cancellation is cooperative. A cancellation arriving after the last check
can race with successful publication; no all-or-nothing cancellation guarantee
is made. Tests cover an already-set token and a FIFO with no writer, not
cancellation during ordinary file copying. The preceding Python manifest
validator can itself block on a substituted FIFO; the Unix open fix applies to
the Rust store only. Source validation and file mutation after verification
remain separate concerns; the MLP/CUTLASS rank-level loaded-buffer digest
described in `STAGE_MLP_LOADED_BUFFER_BINDING.md` remains in effect.

## Evidence

- Linux `cargo test --workspace --locked` and the release Rust/PyO3 build
  succeeded. One Rust test was ignored. Nine focused snapshot tests and an
  atomic no-replace unit test passed, including cases reproduced as failures
  before the fixes. The real PyO3 snapshot smoke test succeeded.
- The complete Python integration suite reported **631 passed, 9 skipped**.
  It includes a test that requires the actual Rust owner and corrupts its
  retained snapshot during runtime preparation; `NativeSession.run` rejects
  the request before submitting it. The first full-suite attempt reported one
  missing notebook because the remote baseline archive omitted it; the
  subsequent run restored the Git-verified notebook and passed. Both logs are
  preserved. Skipped/ignored tests are not claimed as passes.
- Twenty bounded public-session cases passed on one and two RTX 3060s with
  Python and Rust owners, including repeated, unsolved, changed-model and
  changed-beam requests. Four mode-transition groups also passed. The
  independently checked Rust cases reported `model_snapshot.owner="rust"`.
- The independent verifier replays returned paths and checks the per-rank
  loaded-buffer receipts. Its benchmark check compares 45 geometry values in
  both rank logs from each of the 15 measurement sessions; all matched within
  each rank. Reported rank-level `gpu_free_before_bytes` was constant across
  sessions for each GPU. The 66 bootstrap logs are content-verified in the
  portable `rust-mlp-snapshot-3060/bootstrap-receipts.tar.gz`.
- Compute Sanitizer, T4, Hopper, Blackwell, multi-node Slurm, CNOT, LibTorch and
  trained-model quality are outside this stage's hardware acceptance.

The intended baseline was published commit
`30f7d71ec6787621aa7d282aee1ee8a556732d02`, with the exact overlay in
`rust-mlp-snapshot-3060/tested-overlay-manifest.json`. The archived run
confirms the overlay files before and after the checks, but does not contain an
independent final manifest of every file in the remote baseline tree.
The locally retained baseline archive has SHA-256
`67537a557d7d274909a8e4d69e210daf94d489f42a76afa7742cad70aa9b7699`.
All 487 archived files match blobs in the stated Git commit; 439 require
CRLF-to-LF normalization for that comparison. This checks the local archive,
not the final remote tree after extraction.
The overlay includes the missing notebook whose Git blob is
`0fe35bd898f675dc4fa3b99098e81e33da55b815`. The GPU run used CUDA 12.8,
driver 595.99.02 and two 12 GiB RTX 3060s. The complete archived run is
`.local/cache/rust-mlp-snapshot-20261114/evidence.tar.gz`, SHA-256
`7e9e0e2c6f59605409b92ac4e7a970e43fdb0f6924d6d7b7c4d9c59f3f85b5c6`.
All 843 distinct blobs and 3,256 records were verified after transfer; 975
rebuildable files were recorded as omitted. The portable receipt archive has
SHA-256 `3fb7c4ef019fca599e45a2f0e88c66185834baf570a41e4f2814aa6c6b4c79ae`.

## Bounded paired measurement

The 18,369,835-byte synthetic FP16 Hamming model used width 512 and eight
residual blocks, two ranks, logical beam 4096 and six search depths. Each
sample starts a new resident session, warms it and measures a repeated public
call; sessions do not coexist. Five alternating pairs compare the same Rust
session with snapshot reuse off, with the Python snapshot owner, and with the
Rust snapshot owner. This is a warmed process/build/disk, not a cold-process or
first-build measurement, and is not a saturated GPU-step benchmark.

Median repeated full-call wall times were **0.575211 s** without reuse,
**0.566779 s** with the Python owner and **0.576233 s** with the Rust owner.
The median paired change for Rust reuse versus no reuse was **+0.20%**; versus
the Python owner it was **+1.39%**, with individual owner comparisons from
-0.63% to +5.34%. These samples do not establish an improvement. No actual
copy traffic, allocation counts, per-mode peak VRAM or isolated full-step time
was measured; the option remains OFF by default.

Task-owned Vast lease `60658129` was destroyed and its absence confirmed. At
the recorded $0.1348148/hour, the elapsed GPU/disk estimate is approximately
**$0.1689–$0.1712**, excluding network charges and without claiming a provider
invoice. The task-specific guard was stopped after confirmed provider absence.
