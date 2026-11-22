# Experimental stable owner partition

Status: opt-in only. The measured full-step profile did not meet the speed or
memory promotion gate. The normal Python build explicitly selects this path OFF.

## Implementation and ownership

`BEAM_EXPERIMENTAL_OWNER_RADIX=ON` replaces the per-owner full-array passes in
multi-rank Stream 3 with a stable CUB radix sort of owner keys and
`CandidateMeta`, followed by device-side range counting and contiguous gathers.
The key width includes the sentinel after the largest valid owner. Hashes and
original candidate IDs are preserved. Single-rank production dispatch keeps
its existing short path. The partition borrows the existing A/B arrays and CUB
scratch and finishes before the next same-stream borrower; no large arena is
added. The planner and the launch query actual CUB scratch. Insufficient scratch
is rejected before enqueueing work with required and available byte counts.

In this initial measured build, the device guard for a corrupted unique count
above capacity trapped without reporting the required and available counts.
The follow-up implementation and tests for this diagnostic are documented in
`STAGE4_OWNER_COUNT.md`. The measured and regression inputs here satisfy the
upstream count-capacity contract.

## Correctness

The final build snapshot is `green3/source-manifest.json` (SHA-256
`6e03ff12107c446bb4871e1bf1d55fd3655eab9289b7b6993f942ba90d1c8621`).
The owner test covered 1, 2, 3, 5, 8 and 256 **logical** owners, empty and
partial inputs, same-owner and mixed routes, stable ordering, original IDs,
scores, hashes and three graph replays against a CPU oracle: 1,848 checks per
run. The 256-owner test simulates routing domains on one physical GPU; it is
not a 256-GPU run. Stream 3/4, graph, bounded union, large union, static-memory,
stitched, final partial/full-count and 25 Python build tests passed. Repeated
native sessions matched the reference on one and two physical RTX 3060 GPUs.

Compute Sanitizer memcheck reported zero errors and leaks for the owner test on
each GPU and for the Stream 3 and stitched tests; owner initcheck reported zero
errors. With owner radix, Stream 4 index sort and scan all enabled, the owner,
logical-union graph and stitched tests passed memcheck with zero errors and
leaks. These are scoped checks, not multi-node, CNOT, other-GPU-family or
scientific acceptance.

An earlier underallocated-scratch test returned an expected CUB error but also
produced two CUDA API errors under sanitizer (`sanitizer2/`). The final build
queries and validates the size before launching CUB; zero and one-byte-short
allocations then passed the negative test without sanitizer errors. The earlier
logs are retained as failed evidence.

## Paired full-step measurement

Two RTX 3060 GPUs, identical Hamming96 scalar MLP FP16 input and model, logical
beam width 4096 and frozen profile were used. Each of five alternating process
pairs contains one warmup and five complete steps; times use the slower rank.
The binaries differ in the owner option and were built from the same source.

| Throughput change by pair | Median change | Median pair-median step, control / radix | Planned bytes per GPU, control / radix |
| --- | ---: | ---: | ---: |
| +2.64%, -0.22%, -2.56%, +0.44%, -0.20% | -0.20% | 20.240 / 20.282 ms | 64,007,106 / 64,007,106 |

This is a full-step fixture measurement, not cold-start or repeated-session
performance, physical allocation peak or physical memory-traffic measurement.
Sampled telemetry can miss brief excursions. The 465-file raw evidence archive
has SHA-256 `2000c1688b990898d95aadd5cfdedd8baa75733035c08d69ec43a49237aab0bb`;
per-file hashes and build manifests are in `stage4-owner-radix-3060/`.
