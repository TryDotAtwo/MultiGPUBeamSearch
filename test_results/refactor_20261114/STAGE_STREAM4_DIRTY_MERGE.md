# Experimental dirty-run merge for Stream 4

The off-by-default CMake option `BEAM_EXPERIMENTAL_DIRTY_MERGE=ON` adds a
shard-local alternative to sorting all threshold survivors. Stable compaction
retains the existing sorted, unique clean prefix; only the dirty run is sorted
as full `Hash128` plus original compacted-metadata index. A tiled merge joins
both sorted runs, and the existing score/parent/route comparison selects the
winner for each hash. Metadata stays immutable through selection and is gathered
after reduction. The maximum real hash remains valid alongside the invalid
sentinel index. There is no shard top-k or truncation, and the final A/B union
is unchanged. The default build, public options and inference arithmetic are
unchanged.

The two index arrays reuse the existing value workspace; GPU scalar counts borrow
histogram buffers only before histogram production. One stream orders all
readers before scratch is reused. CUB requirements are queried for all dirty
sort bounds and full-capacity reduction. The experimental implementation needs
CUB 2.7 or newer. With `BEAM_STREAM4_BOUNDED_SORT=1`, a conditional CUDA graph
selects the quarter, half or full sort bound from the kept dirty count. Without
it, the static sort extent is full capacity, including the sentinel tail.

## Correctness and build evidence

- On two RTX 3060 GPUs, the independent host/device merge-partition oracle
  passed 85 cases per GPU. An intentional placeholder failed the red test
  before the implementation; the failure and subsequent success are retained.
  The full shard fixture compared ordered survivors and score histograms with
  a CPU `std::map` oracle: 35 cases per GPU at capacity 1025, plus capacity 1
  and 257 tests, clean-only/dirty-only inputs, threshold boundaries, duplicate
  hashes and the all-ones hash. Both ordinary and bounded graph modes ran.
- Ten targeted CUDA test binaries passed for the candidate. Four relevant
  binaries also passed with the dirty and index options combined, and with
  both options disabled. A resident comparison on one and two GPUs ran three
  compatible calls per session without replacing rank processes; returned
  paths were independently replayed. The C++ reference used the control runner;
  Rust used its resident worker from commit `7ae022d9383e31d1ee673f567c39443ad8433b3c`
  and the candidate native library. This does not constitute a rebuild or
  acceptance of all Rust/CNOT paths at the final base commit.
- CUDA 12.8 compiled the two touched CUDA translation units with cubins for
  SM75, SM80, SM90 and SM120. Only RTX 3060/SM86 was run; this is not a full
  project build or physical acceptance on T4, Hopper or Blackwell.
- Final memcheck on the capacity-257 fixture reported zero errors and zero
  leaked bytes. Initcheck on capacity 1 reported zero errors. Unfiltered
  racecheck for the ordinary graph and filtered racecheck for the new merge
  kernel in both graph modes reported zero hazards; ordinary-graph synccheck
  on capacity 1 reported zero errors. These checks are scoped to the fixtures.

## Open sanitizer findings

In the bounded/conditional graph, unfiltered racecheck reported 37 hazards
linking `stream4_mark_valid_unique_indices_kernel` and CUB
`DeviceReduceByKeyKernel`; the fixture still produced the expected values.
An independently built pre-existing index-sort path on the base commit
reported 35 hazards involving the same kernels. Bounded synccheck on capacity
1 reported 256 barrier errors in CUB `DeviceRadixSortSingleTileKernel` and a
launch failure; the previous index path reproduced the same 256-error result.
Neither reproduction establishes whether these are tool limitations or real
bugs. The full logs, exit codes and an initial invalid filter invocation are
preserved. The bounded path must not be considered sanitizer-clean or promoted
until these findings are resolved.

## Matched complete-step measurements

Two independent five-pair series alternated baseline and candidate processes.
Both used two RTX 3060s, a synthetic untrained Hamming-96 FP16 MLP, logical beam
4096, the same input and geometry, `BEAM_B_MICRO=768`, inference microbatch 32,
one shard of capacity 131072, and bounded sorting enabled. Each process included
one warmup and five complete steps; measurements use the slower rank, and
telemetry rejected missing or throttled devices.

The first series had per-pair changes of +8.51%, +2.21%, +1.91%, -2.55% and
-2.65% (median +1.91%). The second gave -1.82%, +2.60%, -3.27%, +3.11% and
+1.56% (median +1.56%). At 25 ms sampling, the second series measured peak
memory above idle of 280 MiB for baseline and 282 MiB for the candidate on
each GPU; sampling can miss short peaks. Planned device memory was identical
at 64,007,106 bytes per rank. Neither the required 5% full-step gain nor a
memory reduction was demonstrated. Keep the option experimental and off by
default. No cold-start, trained-model, CNOT, other-GPU or general speedup is
claimed.

## Provenance and lease

The tested base is `280a7a1731f946ecbcfd8c32ffe47f8abe02430a` plus the eight
source/test overlay files recorded in the archive. The initial measurement and
final overlay differ only in test-driver argument filtering. The measured and
rebuilt runner hashes differ; an exact byte comparison found only four changed
bytes in two occurrences of an `nvcc` temporary symbol name. After normalizing
that identifier, the files match byte for byte. The original and rebuilt hashes
and offsets are retained.

`stream4-dirty-merge-3060/evidence.tar.gz` has SHA-256
`e1ce4689c4a2c74d8387d3036532e14c7bc4dda47181833e6bc100016a073d7f`.
Its verifier checks all 405 archived files, the 494 base source files against
Git, the eight overlays, architecture receipts, sanitizer outcomes, paired
statistics and independent session path replay. The archive contains raw logs,
fixtures, snapshots and commands; large executables are represented by hashes.

Owned Vast lease `60718487` used two RTX 3060s at $0.168889/hour. Evidence was
verified before deletion, and fresh provider inventory confirmed absence. The
destroy CLI exited zero but its output was not valid JSON, so the release
wrapper's parser failed; the subsequent absence check, rather than that parser,
establishes release. GPU/disk cost for the bounded recorded interval is
approximately $0.157–$0.174, excluding network and invoice adjustments.
