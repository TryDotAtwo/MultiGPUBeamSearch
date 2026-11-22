# Size-bound Rust verification of an experimental MLP snapshot

The default-OFF `experimental_reuse_model_snapshot` path can verify the
previously validated private MLP snapshot with Rust before submitting a
resident search. Python still validates the source manifest, tensor sizes and
finite values; Rust streams the exact snapshot with bounded scratch and binds
the digest to the validated manifest and tensor lengths. A PyO3 extension
without the new size-bound preparation method falls back to the complete
Python snapshot and launch validation path.

The legacy digest is a concatenation of manifest bytes, graph hash, tensor
names and tensor bytes. Its digest alone does not identify file boundaries.
Regression tests construct different file partitions with identical digest
input bytes. The sized path rejects changed boundaries both while copying the
source and while verifying the retained snapshot, including the manifest
boundary. Rust checks the size of each opened file and the number of bytes
actually read; its reuse key contains the ordered validated sizes. Python also
checks its actual read count and retains the validated sizes in `PreparedModel`.

Session identity includes immutable copies of the graph center and generators,
its geometry and the validated model sizes. A changed center, generators or
move count therefore releases resident workers even if a manually constructed
contract supplies the old graph hash. This does not prove that a manually
constructed `GraphContract` truthfully declares its own hash; ordinary
`GraphContract.from_graph` computes it from the graph.

Cancellation remains cooperative. An already-cancelled token is tested; this
stage does not prove in-flight PyO3 cancellation or atomic cancellation versus
submission. Checking a path and subsequently loading it remain separate
operations. MLP/CUTLASS ranks additionally bind their loaded host buffers to
the digest, as described in `STAGE_MLP_LOADED_BUFFER_BINDING.md`; the existing
LibTorch path does not have that loaded-buffer binding.

## Correctness and provenance

- On Linux with CUDA 12.8, the full `cargo test --workspace --locked` passed
  with one ignored CPU ABI fixture, the release PyO3/worker build passed, the
  real PyO3 snapshot smoke passed, and the full integration suite reported
  **641 passed, 9 skipped**. The Linux snapshot test binary included 15 tests
  including the Unix FIFO case. Local Windows focused checks had 14 Rust and
  58 Python tests passing; the local full Rust workspace check stopped on the
  existing Linux-only physical history reservation test, and is not a pass.
- Regression tests cover identical concatenated hashes with shifted tensor and
  manifest boundaries, source and retained snapshot changes, a changed graph
  geometry, changed graph contents, in-place mutation of manually supplied
  graph lists, and an old PyO3 store falling back to Python. The red logs retain
  the pre-fix reproductions. CUDA Compute Sanitizer, multi-node Slurm, CNOT,
  LibTorch loaded-buffer binding and trained-model quality are not validated
  by this stage.
- The base is commit `df696b736099330a83e5c2802a7e9bb959cdf3ff`. All 493
  selected baseline files were independently reconstructed from that Git
  commit before packaging and checked after remote extraction. Of these, 48
  match Git blobs byte-for-byte and 445 match after LF-to-CRLF conversion. The
  source manifest binds the exact modified files and records raw and normalized
  LF hashes. After the run, all 493 baseline paths plus new overlay paths
  matched the expected remote hashes. Documentation added after testing is not
  part of that tested source overlay.
- The local full evidence archive is
  `.local/cache/rust-mlp-verify-20261114/evidence.tar.gz`, SHA-256
  `722613913282a016becc21f72ae5ae424c79b933dd396c5e7c1e39b63befa3b5`.
  All 353 content blobs and 1,244 records were checked after transfer; 946
  rebuildable intermediates were recorded as omitted. The portable archive of
  20 rank bootstrap logs has SHA-256
  `d52cf9eafd3a39b0f873cf1b3cb5fee61b5f449e68f2f06b901d8e2967a98af1`.
  `rust-mlp-verify-3060/verify_results.py` independently binds baseline files
  to the Git commit, checks the source overlay and essential evidence against
  the archived manifest, and checks rank receipts, 45 geometry values for each
  rank across all ten sessions, solved-path replay and timing calculations.
  Repeating that verification requires Git, the pinned base commit and the
  tested source overlay. The portable receipt binds the CUTLASS archive hash
  but does not include the CUTLASS archive itself.

## Bounded paired measurement

Both modes use the same Rust-owned snapshot, two RTX 3060 12 GiB GPUs, CUDA
12.8, driver 595.99.02, a synthetic FP16 Hamming MLP of 18,369,835 bytes,
logical beam 4096 and six maximum depths. Each session is isolated and warmed;
five pairs alternate the old full Python launch verification and the new Rust
size-and-hash verification. The initial worker launch still performs a separate
Python validation; the harness counts that check separately. Repeated calls
preserve the same two resident workers within each session. The native runner
uses a Release build with `BEAM_ENABLE_DEBUG=ON` and
`BEAM_ENABLE_DEBUG_LOGS=ON` in both modes.

Median repeated full-call time was **1.400792 s** with Python verification and
**1.345925 s** with Rust verification. The median of paired changes was
**-3.59%**, with individual changes from **-7.17% to -2.13%**. The planned 5%
full-step promotion criterion was not measured; the option remains OFF by
default. This is not a saturated-step, peak-VRAM, allocation-count or
PCIe-traffic measurement, and it does not establish a gain for other GPUs or
trained models.

Vast lease `60698253` was destroyed after the archive was downloaded and
verified; fresh provider inventory confirmed absence. At $0.1688889/hour, the
recorded interval implies approximately **$0.0620–$0.0629** for GPU/disk,
excluding network and without claiming a provider invoice. An earlier attempt
on lease `60669912` never ran the job; its guard observed it stopped and it was
later destroyed. Charging the full advertised $0.1416296/hour across the entire
creation-to-confirmed-absence interval gives a conservative **$0.566** bound
before network, not an invoice or a claim that the machine ran that long.
