# Experimental Stream 4 scans

Status: opt-in only. On the measured profile the full-step speed change was
below the 5% promotion threshold, and planned memory did not decrease. No
physical peak-memory reduction was established for this experiment.

## Implementation and ownership

`BEAM_EXPERIMENTAL_STREAM4_SCAN=ON` independently selects CUB DeviceScan for
exclusive block-count offsets and CUB BlockScan for stable per-block compaction.
The output total remains on the GPU. Inactive lanes of the last partial block
participate with a zero contribution. The scan reads disjoint counts and offsets,
then writes its total after the scan in the same stream. The existing sort,
reduction and scan share the slot's preallocated scratch only in stream order.
The planner queries CUB for the full capacity and the two bounded-graph sizes.
An insufficient scratch allocation fails; the production wrapper reports
required and available bytes. The zero-count helper enqueues a device-side zero.

## Correctness

The final build snapshot is `green3/source-manifest.json` (SHA-256
`fdfca32899bebc9e99a3e658b88facee4b40f07c2ca7337548f9f42599a6c719`), with
all four combinations of scan OFF/ON and index sorting OFF/ON tested. The
original radix experiment stayed OFF. On both physical RTX 3060 GPUs, the scan
test covered zero, 1, partial and multi-block counts through 65,537, changing
inputs across CUDA Graph replay, and rejection of one-byte-short CUB scratch.
The separate large logical-union test covered up to 131,074 input records,
thresholding, empty transitions and graph reuse against the CPU oracle. The
ordinary Stream 3/4, bounded graph, logical union, static-memory, stitched and
final partial/full-count fixtures passed. The Python build tests passed 25 cases.

For scan alone and combined with index sorting, Compute Sanitizer memcheck
reported zero errors and zero leaked bytes in six scoped GPU tests, including
the large union; initcheck reported zero errors in two tests. Racecheck on the
scan primitive reported zero errors and warnings. A separate native resident
session matched the reference across three calls on one and two GPUs. These
checks do not establish CNOT, multi-node, scientific, or other-architecture
acceptance. The expected pre-implementation compile failure and the first large
fixture's compile error are retained in `red1/` and `green2/` respectively.

## Paired full-step measurements

Measurements used two RTX 3060 GPUs, the same Hamming96 scalar MLP, FP16,
logical width 4096 and frozen profile, five alternating pairs, one warmup and
five complete steps per process. Times use the slower rank. The two binaries in
each comparison were built from the same source snapshot and differed in the
scan flag; model, input, geometry, runner hashes and sampled telemetry were
checked. Sampled telemetry can miss transient throttling.

| Comparison | Pair throughput changes | Median change | Median pair-median step, control / scan | Planned bytes per GPU, control / scan |
| --- | --- | ---: | ---: | ---: |
| Default versus scan | +0.80%, -1.75%, +0.30%, +0.28%, +0.07% | +0.28% | 20.254 / 20.202 ms | 64,007,106 / 64,007,106 |
| Index versus index + scan | +1.80%, +0.34%, -0.49%, +0.94%, +2.08% | +0.94% | 19.874 / 19.743 ms | 56,822,722 / 56,822,722 |

These are complete-step fixture measurements, not cold-start, repeated-session
performance, allocation-level peak VRAM, or memory-traffic measurements. The
normal Python build explicitly selects scan OFF, including when a stale CMake
cache exists. Raw outputs and source identities are in `stage4-scan-3060/`;
the 386-file archive has SHA-256
`54a0e02d57436577158502dea9942dfa053998f9d1d698d088e767d0a470000d`.
Six build/test inputs used CRLF on the build host; expanding their LF Git blobs
to CRLF reproduces the exact hashes in all four recorded build manifests. The
remaining checked inputs match the recorded bytes directly.
