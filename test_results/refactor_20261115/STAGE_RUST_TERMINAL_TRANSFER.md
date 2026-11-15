# Rust-validated terminal result transfer

The opt-in public Rust session already parsed each native terminal log, checked
the requested width and rank receipts, and independently replayed every path.
Python then discarded that result and parsed the same log again. The adapter
now consumes the Rust result once, bound to the corresponding run directory.
It rejects a stale request sequence or malformed result and clears any
unconsumed result before another submission. A matching PyO3 extension must
advertise `SEARCH_TERMINAL_RESULT_VERSION=1`; an older extension is rejected
before the Rust session builds its native shared library. The Python session
retains its existing parser. Per-job terminal fields are cleared before
combining metadata for a repeated session, preventing old rank-specific or
collection fields from persisting when absent in the current job.

The result shape and the specialized CUDA inference kernels are unchanged.
The normal `NativeSession()` still selects Python ownership; CNOT's search
loop and its checkpoint scorer are not moved by this change. Rust and Python
still write their existing result files. Python continues model preparation,
public-result assembly and its final path replay.

## Verification

- Two local tests first failed because the Rust result was not transferable.
  The focused tests then passed, including one-shot consumption, directory
  binding, failed/stale request invalidation, old-extension rejection,
  model-snapshot transitions and removal of stale per-rank metadata. The
  complete remote integration suite passed **658 tests, 11 skipped**; the
  release Rust workspace tests and PyO3 build also passed.
- On one and two RTX 3060s, both CUTLASS and LibTorch ran six bounded public
  search scenarios with Python and Rust session owners: first call, compatible
  reuse, unsolved, model change, beam change and collection. For each Rust
  call, the Python parser was disabled during the public call and used only
  afterwards for a full result/metadata comparison. The independent archive
  verifier checked all **48 cases** and replayed **96 returned path entries**,
  including repeated paths and collected solutions, from the saved receipts.
- A separate synthetic terminal corpus produced identical values for five
  valid cases and rejection by both parsers for nine malformed cases. Rust
  additionally rejected a native error marker and an impossible collection
  found-depth that the Python parser accepts.
- These checks cover a synthetic Hamming FP16 MLP with fixed widths and disabled
  autotuning. They do not establish trained-model, CNOT, GNN, Transformer,
  ensemble, multi-node, other-GPU, full cancellation/failure-injection or
  sanitizer acceptance for this stage.

## Paired timing

One two-RTX-3060 Rust session performed a cold call, two warmups and five
alternating pairs of complete six-depth public calls at beam width 4096. The
comparison adds the former extra Python parse after Rust validation; the same
resident session, model, inputs and inference policy are used in both modes.
Median walls were 1.810 s with duplicate parsing and 1.682 s with transfer.
The median of the five paired relative changes was **-1.24%**, with individual
changes from -11.77% to +2.27%. The extra parse itself had a 0.742 ms median,
so the wider wall-time difference is dominated by variation and does not
establish a full-step speedup. The cold call was 6.055 s with an already-built
cache. A single post-run VRAM sample was 301 MiB per GPU; it is not peak VRAM.
No allocation, copy-volume or CPU-wait trace was collected. No default-profile
promotion follows from this experiment.

## Provenance and lease

The tested source was `4fad51d7a9dfccb5938ca1a1249f3d6e2b95f494` plus eleven
overlays; the archive includes the base, overlays, driver, raw logs and
receipts. `rust-session-result/verify_results.py` compares all 533 base files
to Git, checks each overlay and reconstructs results from the GPU logs. The
archive SHA-256 is
`fa39e23fcfebfc251e129e869b37d0f2657edd9b5741cad73374f304fc7ec64f`.

Vast lease `60729483` used two RTX 3060s at $0.136889/hour for GPU and disk.
It was destroyed after archive verification; a fresh provider inventory
confirmed absence and the task-specific guard was stopped. The provider had
reported **$0.054** in GPU, storage and download charges when queried. The
elapsed-time GPU/disk estimate was **$0.0552–$0.0554**, excluding network and
later invoice adjustments.
