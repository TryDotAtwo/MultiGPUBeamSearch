# Runtime requirements and acceptance scope

The migration keeps one active `main` checkout. Archives, builds and caches belong
under ignored `.local/` on D for the local workspace; the approved task executes
compilation and tests on its owned Vast instance. Models, checkpoints, canonical
results and third-party repositories are retained. Raw checks belong under
`test_results/` with source identities and file hashes.

## Toolchain

| Component | Observed build/test environment |
| --- | --- |
| Rust | 1.99.1 and 1.100.0; edition 2021; workspace dependency versions pinned by Cargo.lock |
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

- The optional standalone CNOT domain ABI owns one device, one stream and
  fixed-capacity allocations on its creating thread. It validates ABI version,
  geometry, descriptors, checked sizes and CUB scratch before use, and reports
  capacity errors without truncating. Its host slices remain borrowed only for
  synchronized calls; Rust does not form references to device addresses.
  The standalone domain is not connected to public search or NCCL and does
  not replace the GPU-resident Torch path. A Rust close consumes its handle;
  after a CUDA cleanup/device failure, resource release cannot be retried
  through that handle. See the source-bound results in
  `test_results/refactor_20261117/STAGE_CNOT_OWNED_DOMAIN.md`.
- Its optional device-owner ABI borrows contiguous input and output CUDA views
  on one device for a synchronous call. The caller guarantees physical bounds,
  context accessibility, lifetimes, non-overlap, a valid same-context stream
  and producer ordering;
  the implementation checks declared sizes, address alignment, pointer device
  and stream context. It does not transfer matrices or owners through the host.
  The extension is optional for compatibility with the base ABI; Rust CNOT
  search uses it only when `MGBS_CNOT_OWNED_LIBRARY` is explicitly set on the
  rank-local CUDA call.
- An additive borrowed-expansion operation accepts int64 parent indices and
  int64 or uint16 moves. The caller guarantees valid descriptor values and
  output/input non-overlap; invalid CUDA descriptors retain the existing
  kernel's device-error behavior. The optional device-only context owns its
  thread-bound stream and configured capacities without allocating host-ABI
  stage buffers; host operations on it return an error. The old full context
  and libraries remain supported. Rust CNOT search uses this operation in the
  same explicit opt-in path as device ownership.
- An optional borrowed first-unseen operation preserves full-matrix CNOT
  equality and first-child order. Its caller owns input and output allocations,
  int32 ordinals and count, uint8 flags, and 256-byte-aligned CUB scratch
  sized by an explicit query. Used writable ranges cannot overlap each other or
  the inputs; the call completes the supplied stream before returning. Empty
  batches write a zero device count. No matrix or count is copied to the host.
- In Rust CNOT search, `MGBS_CNOT_OWNED_LIBRARY` explicitly opts each CUDA rank
  into borrowed device expansion, owner assignment and first-unseen selection.
  Each process owns its thread-bound native context. Torch owns all input,
  descriptor, output and scratch allocations; the bridge passes checked local
  device views on the calling rank's current stream and retains the tensor
  handles through synchronous native completion. Rust does not construct
  references from CUDA addresses and no device pointer crosses a rank boundary.
  Existing NCCL transport, CNOT scorer, full-matrix equality and metadata
  assembly are unchanged. The default, CPU and Python paths stay unchanged. A
  selected incompatible library fails through coordinated rank error handling
  rather than silently changing implementation.
- Before this Rust bridge calls the borrowed expansion ABI, it checks parent
  indices and physical move IDs on the device, then reads one boolean result
  from each descriptor tensor on the host. The low-level C ABI still requires
  valid descriptors. These scalar reads turn invalid values into ordinary
  rank-local errors before a device assertion can poison the CUDA context; the
  existing rank-wide stage vote must propagate that error. The two-rank
  injected-error and post-error reuse check is not yet accepted; see
  `test_results/refactor_20261117/cnot-device-error/GITHUB_STATUS.md`.
- A logical Hash128 uses every bit, ordered by high half then low half. CandidateMeta
  remains 32 bytes and aligned to 32. Stream3 retains original candidate IDs.
- Persistent permutation padding stays zero. CNOT packed matrices retain their
  separate geometry and full-word equality checks.
- The opt-in Rust CNOT validator requires 2–256 rows of little-endian packed
  words, zero tail padding, and an invertible GF(2) matrix. Every returned gate
  is a control-major physical ID and construction replay compares all words.
  Each rank must select the same validator; an unavailable extension fails
  closed. CNOT's Rust controller and validator are independent choices;
  arbitrary Python scorers and the existing native checkpoint scorer remain
  supported.
- `search_cnot(engine="auto")` uses the Rust controller when a matching
  extension is installed, and otherwise retains the Python reference. All
  ranks must resolve to the same engine. `engine="rust"` fails closed without
  the controller. The controller owns stage order, history and the global radix
  cutoff passes; CUDA key preparation, radix histograms, mask updates and final
  record ordering use C++ ATen when the matching operators are available.
  The newer rank-local backend retains the cutoff's keys and masks in Rust;
  it keeps the same batch operations, rank collective sequence, and legacy
  Python adapter for older controller/extension combinations.
  The newer stage backend retains rank-local frontier, candidate, closed-set,
  metadata and history tensor handles in Rust while PyTorch owns storage. All
  ranks must advertise that backend; otherwise all use the older Python stage
  adapter. No GPU address is converted to a Rust reference or sent to a rank.
  Other Torch stages, scorer calls and collective management are still
  transitional.
  CUDA local-goal evaluation, candidate expansion, full-matrix distinct
  preparation, winner packing and received-frontier handling use C++ ATen when
  the operator library has matching symbols; older prebuilt libraries keep the
  tensor implementation. The Rust stage backend coordinates the goal and its
  errors using the same rank exchange; PyTorch continues to own all tensor
  storage. The older Python stage adapter retains its previous behavior.
  The Rust backend also coordinates initialization, expansion and distinct
  stage errors with the existing vote and failure-only message gather.
  Its score validation and legacy CPU/older-CUDA expansion and distinct
  sequences are Rust-owned while using the same Torch and domain operators;
  mixed operator availability on ranks must remain valid.
  Multi-rank CUDA send-packet grouping also uses C++ ATen when available;
  stable owner ordering and the existing NCCL transport are retained. The
  Rust stage backend orchestrates that exchange, including the old-operator
  tensor fallback and the existing CPU/Gloo all-to-all sequence.
- Rust-owned CNOT Stream 5 transport requires the existing ABI 1, owned ABI 1
  and exchange ABI 1, a valid native ID and rank geometry, and the same thread
  for initialization and close. After engine admission, all ranks choose the
  Rust transport only if every rank supports it; otherwise all use the existing
  transport. The owned ABI retains the initializing CUDA device and restores
  the caller's current device after cleanup. The native library outlives its
  communicator; dropping an open owner aborts it. Torch owns CUDA buffers;
  FFI validates their device, dtype and shape and records stream use. A failed
  close consumes the handle and reports the error; an underlying CUDA/NCCL
  resource may remain allocated and this version cannot retry that handle.
- CNOT validation, rank identity, the identity fast path and search must use
  the same owned contiguous device snapshot, not a mutable caller alias. The
  caller must synchronize concurrent writes during snapshot creation; an
  atomic cross-stream capture is not provided. The retained copy is at most
  8192 bytes for the supported CNOT geometry.
- A managed Rust permutation session must bind the validated CayleyPy graph hash
  to the actual center and ordered generators loaded by every native rank. An
  unsupported worker or missing ABI extension must fail before search. The
  bound target is the central state; the separate repair mode requires its own
  contract. Names and storage padding are excluded from this graph hash.
- A public Rust session consumes exactly the Rust-validated terminal result for
  the matching run directory. Each result is consumed at most once; a failed
  request invalidates an unconsumed predecessor. Reject an older PyO3 module
  without terminal-result protocol version 1 before building the native library.
  Per-run terminal metadata must not inherit stale rank or collection fields.
- A capable Rust persistent session owns the complete preparation callback outcome metadata paired with the typed preparation identity. Repeated Rust outcomes are directory-bound and consumed once; Rust refreshes the per-run terminal fields and writes the result. Extensions without this optional method retain the Python metadata fallback and existing result contract.
- Unscoped public native searches and `NativeSession()` select the Rust session
  owner when the matching PyO3 API and packaged `beam-runner` are present.
  Missing optional Rust runtime files retain the legacy Python owner; explicit
  `engine="python"` selects it and `engine="rust"` fails closed.
- All-rank admission preserves the exact requested beam; alignment affects physical
  capacities. Buffer overflow reports required/available sizes, never a smaller search.
- Stream4 uses threshold, compaction, fixed-scratch sort/reduce and compaction;
  there is no shard top-k, semantic shard cap or hash-table replacement.
- Experimental index reduction must reject all indices outside the live compacted
  metadata range, including arbitrary values from inactive CUB lanes. Its live
  count and CUB's unique-count output occupy distinct device locations.
- `BEAM_EXPERIMENTAL_DIRTY_MERGE` is a CMake-only, off-by-default experiment
  requiring CUB 2.7 or newer. Its stable compacted clean prefix must be sorted
  and unique, the dirty run is sorted by the complete Hash128 and merged with
  the prefix before the existing winner reduction. Metadata stays immutable
  until gather; sentinel indices are invalid even for the maximum real hash.
  CUB temp sizes cover each conditional bound and the full reduction. With
  `BEAM_STREAM4_BOUNDED_SORT=1`, the device chooses a bound for the dirty count;
  overflow fails rather than silently omitting candidates. The final A/B union
  remains separate. Conditional-graph sanitizer findings remain unresolved;
  see `test_results/refactor_20261114/STAGE_STREAM4_DIRTY_MERGE.md`.
- Experimental Stream4 scans use disjoint input counts, output offsets and total;
  each block participates fully in BlockScan, including inactive tail lanes. CUB
  scratch is queried before execution and is reused only in stream order. An
  insufficient allocation must fail with required and available byte counts.
- Experimental Stream3 owner radix keeps original candidate IDs, stable order
  within each owner and separate contiguous local/remote outputs. Its owner key
  width includes the sentinel for every supported world size. The preallocated
  partition buffers and scratch cannot be reused until routing readers complete.
  One-GPU dispatch retains its existing short path.
- Experimental bounded score histogram sorts include all bits of the sentinel;
  launch and CUB scratch queries use the same range. Hash identity remains 128-bit.
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

