# Owner partition capacity diagnostics

Status: the owner-radix option remains experimental and OFF by default. This
change closes the out-of-range device-count diagnostic gap in the initial
implementation described in `STAGE4_OWNER_RADIX.md`.

Before routing, a same-stream device check compares the unique count with the
candidate capacity. If it is too large, the existing fatal trace receives code
3003, the required count and the available count. Routing emits zero counts and
does not write candidate outputs; the dispatcher checks the fatal flag before
the exchange and formats the two sizes in the error. An earlier fatal trace is
not overwritten. This prevents silent truncation and avoids a device trap for
this count-capacity error. The scratch-size check remains a separate host-side
required/available error before enqueue.

## Scope of verification

The build snapshot `green1/source-manifest.json` has SHA-256
`93f4b3536a182c82706798a05eef47f61a1eaf6a2dc16d9c103ab020a77e7872`.
On each physical RTX 3060 the owner test injected counts of capacity + 1 and
`UINT32_MAX` into one- and two-owner paths, in eager execution and captured
graphs. It verified the fatal code and exact device trace values, zero published
counts and offsets, untouched output canaries, preservation of an earlier fatal
trace and continued execution of 1,848 normal routing checks. These logical
owner cases are not physical multi-rank failure injection.

Memcheck reported zero errors and leaked bytes on both GPUs; initcheck reported
zero errors for the owner test. Stream 3 and stitched memcheck passed. A
combined owner/index/scan build passed its regression suite and scoped memcheck
for the owner, logical-union graph and stitched tests. The default-off build
also passed the listed CUDA regression tests. Repeated native sessions matched
the reference on one and two GPUs. The dispatcher formatting and placement of
the fatal gate were reviewed in source; an injected end-to-end multi-rank error
and the error text through the public API were not tested in this stage.

## Paired full-step measurement

Five alternating process pairs on two RTX 3060 GPUs used identical Hamming96
scalar MLP FP16 input, width 4096 and frozen profile, with one warmup and five
complete steps per process. Times use the slower rank. Pair throughput changes
were +2.17%, -0.13%, -3.56%, +1.72% and -2.05%; the median was **-0.13%**.
Median pair-median step times were 20.262 ms control and 20.300 ms owner radix.
Both plans reported 64,007,106 bytes per GPU. The result does not meet the
promotion threshold; no cold-start, repeated-session performance or physical
peak-memory/traffic claim follows from it.

The 277-file evidence archive has SHA-256
`4e7186adddef1ac8684f1a6051eaf12e8d3fec2deb546da1a484c49408229bbc`;
raw logs, source identities, inputs and per-file hashes are under
`stage4-owner-count-3060/`. The expected pre-implementation compile failure is
retained in `red1/`.
