# Experimental MLP loaded-buffer binding

The opt-in session snapshot previously validated files before the native loader
reopened them. A concurrent modification could therefore change the bytes
loaded by a rank after validation. For explicit MLP `NativeModel` requests using
`backend="native"` and CUTLASS, each new rank now hashes the exact manifest
bytes used for parsing and the already-owned host tensor vectors before GPU
upload. The SHA-256 includes the graph hash and lexicographically ordered tensor
names, matching the Python artifact digest. Hashing uses bounded 64 KiB input
chunks and does not reread the weights or make a second full tensor copy.

The controller requires one matching startup receipt per rank before returning
the first result. A compatible repeated call reuses the already loaded weights
and initial verification; it does not rehash GPU memory. Worker identity
includes whether this binding is active, so a prior unbound `backend="auto"`
group cannot be silently reused after switching to strict native mode. An old
external runner that does not emit the receipt is rejected before returning a
result; it can nevertheless execute its first search and write private history
before that rejection. This is a result-level check, not a side-effect barrier.

The ordinary path, LibTorch, ensembles and CNOT are outside this binding. The
option remains off by default and this Python-owned snapshot is an interim
step toward Rust-owned preparation. Calibration processes can also hash the
weights when the opt-in is active; cold and calibration overhead needs separate
measurement.

## Correctness evidence

- The host-only C++/Python oracle exercises FP16, BF16, LayerNorm, twelve
  residual blocks, lexical ordering, buffers above 64 KiB, invalid hashes,
  missing hash pairs, file mutation, and mutation of already loaded vectors.
  The initial independent CPU test reproduced the old acceptance of changed
  weights; the revised test passed locally and on a private CPU notebook.
- On Linux, the host CTest passed and the integration suite reported **627
  passed, 9 skipped**. Twenty focused mocked receipt and mode-transition cases
  also passed locally. The skipped tests are not claimed as passes.
- On one and two RTX 3060 GPUs, 20 bounded public-session cases passed for
  Python and Rust owners: first, repeated, unsolved, changed-model, and
  changed-beam calls. Four additional groups exercised
  `auto → native → native → auto` and checked worker replacement/reuse.
  Returned paths were independently replayed, and the archived per-rank
  receipts were checked against the recorded model hash.
- These checks do not cover full application Compute Sanitizer, CNOT, trained
  model quality, T4, Hopper, Blackwell, or multi-node operation.

`mlp-binding-3060/verify_results.py` independently checks the paths, rank
receipts, source overlay and paired timing arithmetic. The portable
`bootstrap-receipts.tar.gz` contains the 40 actual rank startup logs with
content hashes; reused calls point to the original workers' logs. Its SHA-256
is `bac03ee3f516a21fb6bf285afbee06ce301fdb220ede58be31757f997fd1e920`.
The full locally preserved evidence archive is
`.local/cache/mlp-binding-20261114/evidence.tar.gz`, SHA-256
`8ff9206432a538451e7af182d64cefe19834a5beecf9ed4202b61f345a00cd00`.
All 542 distinct content blobs and 1,761 manifest records were verified;
962 rebuildable cache/intermediate files were recorded as omitted. The archive
stores actual files rather than repeated-call symlinks. Four representative
files were restored and checked. Compact logs, results and manifests are in
`mlp-binding-3060/`.

The source baseline was published commit
`65a1952c265bd60c6d69a7ee90687c0a7d336888` plus the exact overlay recorded in
`tested-overlay-manifest.json`. The manifest also records LF-normalized hashes
because Git normalizes CRLF source files. The Rust worker and extension were
reused from the previous verified build after confirming their Rust source was
unchanged; the CUDA runner and library were rebuilt for this overlay. CUTLASS
was archived from Git commit `afa1772203677c5118fcd82537a9c8fefbcc7008`;
the extracted source digest was
`c6a6fa64a544036c6ecd9a9875e6985d456f4186a8aeb9a82e0132c3897a4645`.
This clean Git archive differs from the previously used CUTLASS cache digest
`0adcc47dbf4082dc2acbf0bd07545aace5a63dcab8926f919a2e4b236d314e56`;
the reason for that difference was not established, so it is not presented as
byte-identical to the previous cache. The first bootstrap stopped at that
digest check; the continuation verified the archive-derived digest and ran the
tests. Driver 580.173.02, CUDA 12.8 and 12 GiB per RTX 3060 are recorded in
the accompanying receipts.

## Limited performance measurement and cost

For an 18,369,835-byte synthetic Hamming FP16 MLP (width 512, eight residual
blocks), two RTX 3060s, beam 4096 and six depths, the median of five alternating
paired repeated full calls was **0.848941 s baseline** and **0.829958 s with
snapshot reuse and binding**. The median of the paired wall-time changes was
**-2.91%**. Individual changes ranged from -3.11% to -1.24%. This does not
meet the 5% speed-promotion threshold. Median worker timers were 0.306004 s
and 0.305887 s; they are not isolated CUDA timings. One cold call per mode
measured 3.220 s and 3.489 s respectively and is insufficient to characterize
cold-start or calibration overhead. No isolated saturated-step timing, measured
copy traffic or per-mode peak VRAM was collected.

Task-owned Vast lease `60654136` used two RTX 3060 GPUs and was destroyed;
provider absence is recorded in `mlp-binding-3060/release-receipt.json`.
At the recorded $0.14163/hour rate, the interval between the provider's last
duration snapshot and verified absence gives approximately **$0.0646–$0.0658**
for GPU and disk, excluding network charges and not a provider invoice. The
task-specific watchdog was stopped after verified absence.
