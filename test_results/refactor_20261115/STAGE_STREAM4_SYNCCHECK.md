# Stream 4 conditional-graph sanitizer investigation

The published Stream 4 implementation was not changed. This stage adds oracle
cases for empty and one-item inputs, the real all-ones `Hash128`, its duplicate
across clean and dirty runs, and a one-item final logical union. The shard cases
also cover both relative orders of an all-ones hash and a lower hash.

The exact final test sources were compiled against the published index-sort
and dirty-merge CUDA implementations. On two RTX 3060 GPUs, ordinary and
bounded graphs passed the CPU oracle at capacities 1, 2, 3, 4, and 257: 20
process/graph combinations, each checking both devices. Bounded-graph memcheck
passed at capacities 1 and 257 for both implementations. A separate index-sort
logical-union test, including a valid one-item all-ones key and replayed graphs,
passed direct execution, memcheck and synccheck on both devices. These are
fixture results, not end-to-end or multi-node acceptance.

## Investigation and limits

CUDA 12.8.93, Compute Sanitizer 2025.1.0.0 and driver 595.104.02 reproduced
the original bounded Stream 4 capacity-1 finding: unfiltered synccheck reported
256 divergent-barrier errors in the CUB `Hash128` radix sort and the application
reported an unspecified launch failure. The ordinary graph passed synccheck;
both graph modes passed direct execution and memcheck in this fixture.

An uncommitted candidate skipped the radix sort for one item and changed which
existing buffers were the disjoint reducer/merge inputs and outputs. Direct
CPU-oracle runs passed on both devices for capacities 1, 2, 3, 4 and 257 in
ordinary and bounded modes. It did **not** resolve the finding: on the bounded
capacity-1 graph synccheck instead reported 256 barrier errors in the dirty
merge kernel and 128 in CUB ReduceByKey for the index variant. Bounded
capacity-257 synccheck still reported 256 errors at the radix sort. Unfiltered
racecheck also reported hazards. The candidate has no measured complete-step
benefit and was removed from the working implementation; its source overlay and
logs remain archived for reproduction. The moved warning is evidence about the
scope of the symptom, not proof of a sanitizer defect or of application safety.

Three source-pinned, separate minimal comparisons used the project's full-width
radix sorter, CUB ReduceByKey, a shared-memory marker, and ordinary versus
conditional graphs. The first tested one conditional node and three chained
nodes with either the first or last body selected. The second added a separate
block-synchronizing parent-graph kernel before the conditional selector. The
third put a partially active 256-thread kernel immediately before the sort,
with 1, 65 or 256 threads doing work. For sizes 1 and 1025, all direct and
synccheck runs passed; the recorded memcheck cases passed as well. Thus those
individual structural features were insufficient to reproduce the production
warning in these examples. The precise cause remains unknown, and neither the
bounded path nor its racecheck warnings are certified clean. The existing
experimental options remain off by default.

## Provenance and cost

The base commit was `26fb7ed824e17a78d0b286f7197e4faa647f6254`. The archived
source tar was compared file-by-file with 192 blobs from that commit, allowing
only the Windows Git checkout's CRLF conversion. The tested changed test files
and the discarded candidate overlay are separately SHA-256 bound. `verify.py`
checks the source and initial reproduction; `verify_extra.py` checks the
candidate, minimal controls, final test sources, raw exit codes and expected
sanitizer outcomes. Reports preserve both successes and failures. The archive
hashes are recorded in `stream4-synccheck/verification*.json`.

Vast lease `60742420` failed to complete image loading and was deleted without
running tests. Lease `60743283` used 2× RTX 3060 at $0.13689/hour. Both were
removed, their absence independently checked, and their own stop guards ended.
At query time, provider-reported charges for the two leases totaled **$0.075**;
later invoice adjustments are possible. No other hardware was tested in this
stage and no performance gain is claimed.
