# One-time CUB scratch initialization

The static GPU arena now initializes the union of its allocated CUB scratch
ranges before it is returned. These are Stream 3, every Stream 4 sort slot,
the final logical union, and the final-materialization range when present.
The host merges overlapping ranges, so phase overlays are initialized once.
The initialization stream is nonblocking and explicitly synchronized before
execution streams can use the arena. Neither the whole arena nor any range on
each search step is cleared. A failed initialization releases the allocation.
Direct low-level callers providing scratch themselves retain responsibility for
initialization and lifetime.

## Reproduction and scoped checks

The unmodified production allocator, with the new tests applied, produced
2,248 initcheck API-access reports when the test read the arena scratch extents
and 9,912 reports inside CUB during the Stream 4 histogram/reduction test using
the production arena pointer (`red2/`). The CPU histogram oracle still matched;
that alone does not establish that uninitialized reads are harmless.

With the fix, `green1/` shows zero initcheck errors on each physical RTX 3060
for: both single- and two-rank *layouts* before blanket initialization; the
Stream 3/4 histogram oracle using real arena scratch through eager and repeated
graph launches; and the final-union primitive using the actual arena with only
its live histogram inputs separately initialized. The score test also passed
memcheck on each GPU with zero errors and leaked bytes. Stream 3, Stream 4 and
stitched CUDA regression tests passed. The layout test's multi-rank layout is
not a multi-GPU collective.

`combined2/` repeats the scoped checks with all five experimental options
enabled simultaneously: full Hash128 radix, Stream 4 index sorting, Stream 4
scan, owner radix and bounded score bits. It includes an index-reducer check
that actually sees both GPUs and radix checks on each GPU. The preceding
`combined1/` failed because the index-reducer test requires two visible GPUs
and was launched with only one visible; no sanitizer error was reported.
The earlier `red1/` records an invalid test geometry, corrected in `red2/`.

Repeated compatible Rust/native sessions matched the reference on one and two
physical GPUs in both default and combined builds. This is a small Hamming
fixture, not validation of CNOT, multi-node operation, other GPU architectures,
every precision/model, or a whole-application initcheck. In particular, the
low-level independently allocated test scratch is not automatically initialized
by this change; tests using it must initialize it themselves when required.

## Paired complete-step measurements

Identical Hamming96 FP16 MLP inputs, width 4096, geometry, driver and rank
count were used on RTX 3060 with five alternating independent process pairs;
each process used one warmup and five full steps. All five experimental flags
were OFF. Production source manifests differ only in `cuda/static_memory.cu`
and its header. Runner hashes, plans, input and model hashes, per-rank logs and
telemetry are in `session-pairs1/`.

| Physical GPUs | Pair throughput changes | Median paired change | Median process wall, control / fixed | Planned bytes/GPU |
| --- | --- | ---: | ---: | ---: |
| 1 | +0.58%, +0.28%, -0.03%, +0.29%, +1.83% | +0.29% | 1.056 / 1.002 s | 64,236,482 / 64,236,482 |
| 2 | +2.26%, -0.06%, -0.25%, -3.53%, +1.84% | -0.06% | 1.756 / 1.724 s | 64,007,106 / 64,007,106 |

Process wall includes process creation, model loading, initialization and the
measured search. It is not an isolated CUDA initialization measurement, a cold
GPU/driver boot, or repeated-session performance. These results support no
general speedup claim; the initialization is a correctness/diagnostics fix.

The locally retained 410-file raw archive SHA-256 is
`da370eb04350cef3495b95c5b6e9bf5da394a570833a0066b27941e5c69f4f10`.
The 22 generated model files are retained locally and omitted from publication;
their sizes and hashes are recorded separately in the manifest. Nine
supplemental command/config files, the ignore rule, and hashes of the 398
published evidence files are listed in `stage4-cub-init-3060/manifest.json`.
CUDA binaries are identified by hash but omitted from the archive; prior
score-bits evidence retains the control's source snapshot and relevant logs.
