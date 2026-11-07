# Experimental Stream 4 index sorting

Status: opt-in only; the default remains comparison sort. The independent
Stream 4 measurement showed lower sampled memory without a median slowdown on
one profile. The public build currently chooses its binary before the final
beam/profile is known, so it cannot yet safely select this result only for the
verified profile. Full public-path acceptance remains pending.

## Ownership and implementation

`BEAM_EXPERIMENTAL_INDEX_SORT=ON` independently selects this Stream 4 path;
`BEAM_EXPERIMENTAL_RADIX128` separately controls Stream 3 and the non-index
Stream 4 hash sort. The index path compacts metadata once, sorts full Hash128
keys with `uint32` indices, reduces those indices with the existing score,
parent and route ordering, and gathers winners to the separate survivor arena.
The two disjoint index arrays occupy the first `8 * physical_capacity` bytes of
the existing value-B workspace. The full 32-byte-per-item workspace remains
allocated in this experiment. No extra hot-path allocation was introduced.

The logical sort pair changes from 16-byte key plus 32-byte metadata to 16-byte
key plus 4-byte index. This is a description of the logical pair, not a measured
GPU traffic figure. CUB scratch sizes are queried for the chosen value type and
capacity; the arena is not resized from a guessed byte count.

The initial implementation checked only the `UINT32_MAX` sentinel. Compute
Sanitizer found invalid metadata reads: CUB also evaluates the reduction
operation for inactive lanes with other payload values. That failed run is
retained under `validation1/sanitizer`; its earlier paired timings are invalid
for acceptance. The fixed reducer checks the live compacted count, bounded by
the sort capacity, before every metadata dereference. The compacted count stays
unchanged throughout reduction. CUB writes its unique count into the no-longer-
needed block-offset buffer; the next kernel finishes reading it before the scan
reuses that buffer. A device trap guards a compacted count exceeding the sort
bound. No GPU pointer is shared between processes.

## Correctness and memory checks

The initial fixed source snapshot is `green2/source-manifest.json` (manifest
SHA-256 `674e115444e8ff07e1beb3b6e5778328242b0f932b862b26342cba8e95cff43d`).
The final independent-switch build is `green3/source-manifest.json` in
`stage4-index-sort-independent-3060/` (manifest SHA-256
`0f2155a0e03fed9a69e14f46113654ffcc5fd1b6b0b73c4bf1db643e704b11a0`). On two
physical RTX 3060 GPUs, the opt-in build passed the dedicated index reducer,
full-hash radix, Stream 3, Stream 4, logical union, captured and bounded graph,
static-memory, stitched pipeline and final partial/full-count tests. The union
tests compare winners and histograms to a CPU oracle and check that compacted
metadata remains stationary, including empty batches, independent hash halves,
an all-ones hash, duplicate scores, parent/route ties and graph replay.

Compute Sanitizer memcheck with API errors enabled and full leak checking
reported zero errors and zero leaked bytes in five tests, including both graph
union variants and bounded Stream 4 graphs. Initcheck reported zero errors in
the reducer and Stream 4 tests. A separate resident-session comparison against
the reference runner matched paths for three consecutive calls on one and two
GPUs, with fixed beam width 64 and unchanged worker identities. The ordinary
build with both experiments OFF passed the corresponding CUDA regression set;
the Python build tests passed 25 cases. The final source was also built and
tested with index-only, combined radix/index, and both options OFF. The
index-only build again passed memcheck with no errors or leaked bytes, initcheck,
and the repeated one/two-GPU session fixture. These are scoped fixtures, not
complete scientific, CNOT, multi-node or cross-architecture acceptance.

## Paired full-step measurement

Both comparisons used two RTX 3060 GPUs, the same Hamming96 scalar MLP, FP16,
logical width 4096, frozen native CUDA Graph profile, five alternating pairs,
one warmup and five complete steps per process. The slower-rank step time is
used, model/input/binary hashes are checked and sampled GPU telemetry must be
complete and unthrottled. Telemetry can miss transient throttling. These timings
do not measure actual memory traffic or resident-session performance.

The following first comparison predates independent switches and enabled
Stream 3 radix along with Stream 4 index sorting:

| Variant | Median of pair medians | Planned required bytes per GPU |
| --- | ---: | ---: |
| Default comparison sort | 20.277 ms | 64,007,106 |
| Radix keys plus metadata | 23.723 ms | 64,372,674 |
| Radix keys plus indices | 22.951 ms versus default; 22.936 ms versus radix | 56,822,722 |

Relative to default, the corrected index variant had median paired throughput
change **-11.66%** (pairs: -9.08%, -11.78%, -10.02%, -12.90%, -11.66%).
Relative to the full-metadata radix build it was **+3.43%** (pairs: +5.77%,
+3.43%, -0.57%, +1.84%, +3.62%). Since the default comparison also changes Stream
3 sorting, that comparison does not isolate the effect of Stream 4 alone. The
planned memory difference versus default is 7,184,384 bytes per GPU; it is not
a measurement of peak VRAM. The slowdown exceeds the allowed 3% for the memory
promotion criterion.

With Stream 3 held at its production comparison sort, the independent Stream 4
index build produced pair changes of **+2.15%, +2.46%, +3.04%, +1.57%, and
-0.92%** (median **+2.15%**). Median pair-median full-step times were
20.150 ms for the default and 19.726 ms for index sorting. A second five-pair
series sampled whole-device VRAM every 25 ms; changes were **+3.60%, +2.24%,
+5.23%, -2.70%, and -1.74%** (median **+2.24%**). Each pair showed an 8,388,608
byte lower sampled peak above its baseline on each GPU (291,569,664 vs
283,181,056 bytes). The monitoring process observed the same device baselines
before and after each run, with no compute processes at those boundaries. NVML
sampling can miss peaks shorter than the interval; this is measured
whole-device usage, not allocation-level accounting or a traffic measurement.
The 7,184,384-byte planned difference and the sampled 8 MiB difference are
different quantities.

Raw logs, source and binary hashes, telemetry, the failed sanitizer run and all
intermediate measurements are under `stage4-index-sort-3060/`. The 564-file
retrieved archive has SHA-256
`e0924e7ddb25c44cfa4eb729916fe681a7db6c129982aa008b46400d2677e921`.
The independent-switch archive contains 408 files and has SHA-256
`ad90297ed24141f3dbc173557d061f8b0184df767b6806dc57d39490b852a4cc`.
It includes the two initial memory-monitor diagnostics, the corrected measurement,
source manifests, correctness logs, and both full-step comparisons.
Six build and test inputs used CRLF in the Windows checkout and on the remote
build host; their Git blobs use LF. Expanding the staged blobs to CRLF reproduces
the exact SHA-256 in each build manifest. The remaining checked source blobs
match the recorded build bytes directly.
