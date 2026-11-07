# Runtime requirements and acceptance scope

The migration keeps one active `main` checkout. Archives, builds and caches belong
under ignored `.local/` on D for the local workspace; the approved task executes
compilation and tests on its owned Vast instance. Models, checkpoints, canonical
results and third-party repositories are retained. Raw checks belong under
`test_results/` with source identities and file hashes.

## Toolchain

| Component | Observed build/test environment |
| --- | --- |
| Rust | 1.99.1; edition 2021; workspace dependency versions pinned by Cargo.lock |
| Python/PyO3 | Python 3.11; PyO3 0.24; CPython 3.10+ stable ABI |
| CUDA | 12.8, Linux development image |
| PyTorch | CUDA-enabled 2.9.0 for the optional LibTorch and existing CNOT provider |
| NCCL | 2.27.5 |
| CUTLASS | 3.8.0, commit afa1772203677c5118fcd82537a9c8fefbcc7008 |
| Physical GPU checks | one/two RTX 3060, SM86, 12 GiB per GPU |

This table records observed configurations. It does not establish an older Rust
minimum, a CUDA-version compatibility range or physical T4/Hopper/Blackwell
acceptance. CPU-only Rust builds do not link CUDA. CUDA libraries load in owned
rank processes after their ABI/layout and specialization checks.

## Contracts

- A logical Hash128 uses every bit, ordered by high half then low half. CandidateMeta
  remains 32 bytes and aligned to 32. Stream3 retains original candidate IDs.
- Persistent permutation padding stays zero. CNOT packed matrices retain their
  separate geometry and full-word equality checks.
- All-rank admission preserves the exact requested beam; alignment affects physical
  capacities. Buffer overflow reports required/available sizes, never a smaller search.
- Stream4 uses threshold, compaction, fixed-scratch sort/reduce and compaction;
  there is no shard top-k, semantic shard cap or hash-table replacement.
- Experimental index reduction must reject all indices outside the live compacted
  metadata range, including arbitrary values from inactive CUB lanes. Its live
  count and CUB's unique-count output occupy distinct device locations.
- Experimental Stream4 scans use disjoint input counts, output offsets and total;
  each block participates fully in BlockScan, including inactive tail lanes. CUB
  scratch is queried before execution and is reused only in stream order. An
  insufficient allocation must fail with required and available byte counts.
- Reuse requires completion of every reader. GPU-source and pinned-host-history
  reader completion are distinct. Cross-process messages contain no device pointers.
- Only explicit native capacity rejection bounds a capacity search. Rank failure,
  malformed receipts, timeout and cancellation close owned processes and propagate.
- Independent inference probes can restart a failed rank while retaining healthy
  ranks. NCCL planning/search cohorts still fail together. A quality failure is
  never used as an inference memory bound. Every accepted inference candidate
  requires complete rank identity, at least five positive finite timings,
  numerical-oracle success and complete unthrottled GPU telemetry.
- Public options have a strict Rust representation, including integer/duration
  bounds and incompatible option combinations. Python only serializes descriptors
  at the new coarse adapter. Model export and its semantic checks stay in PyTorch.
- A cached inference profile requires complete source/device-bound receipts and
  recomputed timing statistics. Changed model/helper bytes invalidate the request;
  released memory pressure invalidates a capacity-limited cached profile.
- Notebook SIGINT must cancel the owned coarse job and reap its workers before
  KeyboardInterrupt is propagated. Normal probe cleanup must not cancel its parent
  before the verified result can be published.

## Performance and switching

Downstream component timings only nominate profiles. Keep raw all-rank receipts,
frozen inference microbatch and the exact row/parent conversion for scalar MLPs.
Every candidate and every cache hit needs current native memory admission.
Cached full-step results require the same physical geometry and complete timing
cohort. A bounded shortlist is not a global optimum guarantee. Preserve native
required/available diagnostics when an admitted profile overflows during execution;
never reduce the requested beam implicitly to complete calibration.

Preserve inference arithmetic while moving control. Numeric-width admission may
replace only a complete verified measured microbatch and must preserve the exact
frontier. Symbolic max keeps memory, full-step and efficiency evidence separate;
changed inference reserves require readmission. A multiset upper bound is not a
reachable-orbit certificate. A working-width reference uses the same execution
profile, a fresh complete-step cohort and saturated per-rank batches. At most
three refinement/comparison rounds may run; uncertainty/failure must propagate.
CNOT control and public launch switching need separate comparisons. CPU or source
checks do not establish GPU acceptance.

Experimental CUDA sort variants must remain explicit build choices until the
matched full-step promotion gate is met. The public automatic build must select
the accepted default even when a build directory contains an older CMake cache.

Promote a CUDA variant only for a measured profile with identical models, inputs,
logical beam, precision and rank count: warmup plus at least five paired complete
steps; at least 5% time improvement, or measured memory reduction with no more
than 3% slowdown. Keep cold launch, resident calls, VRAM, allocations, copies and
CPU waiting separate. Component timings only propose trials. Missing telemetry
and unstable measurements cannot produce an accepted default.

The authorized total is USD 50 including associated costs. This task may operate
only its own leases. Preserve billing/lease records and stop owned paid resources
when their work ends; do not infer authorization over another project's instances.
