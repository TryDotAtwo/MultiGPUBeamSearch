# Session model snapshot reuse and lifetime

Status: experimental default-OFF Python-owned session cache implemented and
tested on one synthetic two-RTX-3060 profile. A full-call speedup was not
established. Rust still does not own model preparation. This stage does not
change inference precision or GPU search kernels.

Only an explicit MLP `NativeModel` in strict native mode is eligible. Source
and retained snapshot are validated for each eligible call. A compatible,
already-loaded worker can reuse the previous snapshot. A new or incompatible
worker receives a fresh snapshot in its own run directory. The option reports
logical copied artifact bytes, not measured disk traffic. Validation performs
additional reads and allocations and may erase any wall-time benefit.

Public calls reserve the session through preparation, execution and result
assembly. Direct `run`, `launch`, model-cache and NCCL-cache entry points reject
foreign concurrent use; failed direct launches release only workers created by
that launch. The cache binds a reload to the exact prepared model of the current
cache hit. A public result's mutable manifest cannot change the next validated
manifest. Independent review also identified and checked the lock handoff when
another thread has acquired the outer lock but has not reserved the session.

`model-snapshot-validation-final.log` records **91 passed** targeted CPU tests,
including real thread barriers, existing dispatch/reflection cases, model
identity, session lifetime, and corruption between preparation and submission.
The revised cache reconstructs its private manifest from the verified source,
checks that source twice and validates the retained snapshot once in the actual
session before submission or a worker relaunch. The deliberately failing logs include
`model-snapshot-binding-red2.log`, `model-snapshot-helper-race-red.log`, and
`model-snapshot-contender-red.log` and `model-snapshot-validation-red.log`.
The initial `model-snapshot-red.log` could
not create its missing temporary parent and is not implementation evidence;
`model-snapshot-red2.log` is the subsequent test-first run. A broader local
collection in `model-snapshot-cpu-final.log` failed because this Windows Python
environment's installed CayleyPy lacks `ModelConfig`; that suite was not claimed
as passed locally.

The first hardware run used commit `a3b6e8b0dd0e2c2ba8a83ff708578523e9f1a09f`,
two RTX 3060 GPUs, driver 580.173.02, CUDA 12.8 and a synthetic MLP. It reused
previously verified native binaries after checking the complete native and
CUTLASS source digests. Linux Python tests reported **605 passed, 9 skipped**.
Twenty bounded cases covered one/two GPUs, Python/Rust session owners, first,
repeated, unsolved, model-change and beam-change calls. Returned paths were
replayed independently; worker replacement and snapshot copying were checked.

On an 18,369,835-byte synthetic Hamming FP16 model (width 512, eight residual
blocks), five alternating repeated calls at beam 4096 and six depths gave median
full-call wall times **0.818561 s baseline** and **0.866151 s reuse**; the
median of the five paired relative changes was **+4.59%**, a slowdown. Median
worker timers were 0.305955 s and 0.305637 s, respectively; these are not CUDA
kernel timings. The experiment avoids logical snapshot copies on compatible
reuse but performs additional validation reads; it does not meet the promotion
criteria on this fixture.

The first evidence archive is `.local/cache/model-snapshot-20261114/evidence-v1.tar.gz`,
SHA-256 `0935555814637bdebc474a59a14f602248a2026a94248affedc5361e24852d2e`.
All 408 content blobs and 1,308 manifest records were verified locally and four
samples restored; 14 rebuildable items were recorded as omitted. Compact
receipts and raw results are in `model-snapshot-3060/`.

## Revised validation path

Commit `46c02ab50284ebe693141af8637066a7f716d054` was tested on the same
two GPUs, using the same model and public-call fixture. The source and CUTLASS
digests still matched the verified native build; Python integration files and
the binary/worker hashes were recorded before and after the run. The full Linux
suite reported **607 passed, 9 skipped**. The same 20 bounded cases passed on
one and two GPUs with Python and Rust session owners.

The revised path reads the retained snapshot once instead of twice on a cache
hit, and still checks the source twice. Five alternating paired calls produced
median wall times **0.818708 s baseline** and **0.824459 s reuse**. The median
paired change was **+0.58%**; individual changes ranged from -1.72% to +4.53%.
Median worker timers were 0.306277 s and 0.305964 s; they are not isolated CUDA
kernel timings. The fixture recorded logical file bytes, not actual disk traffic,
per-mode peak VRAM, or isolated saturated-step timings. It does not satisfy the
promotion criteria; the flag remains off by default.

The second archive is `.local/cache/model-snapshot-20261114/evidence-v2.tar.gz`,
SHA-256 `96da1fa1467437feeb3aaaffcf741719d3e9b7aaa728584d6b8235a748bd327f`.
All 694 blobs and 2,525 records were verified locally and four samples restored;
28 rebuildable items were recorded as omitted. The independent
`model-snapshot-v2-3060/verify_results.py` replays every recorded solution,
recomputes the five paired changes and checks recorded inputs against the Git
commit and verified archive.

Task-owned Vast lease 60648840 was destroyed and provider absence confirmed;
`model-snapshot-v2-3060/release-receipt.json` records its identity and duration.
The duration/rate estimate is **$0.0376** for GPU and disk, excluding network
charges and without claiming to be the provider invoice.

The existing native loader reopens mutable files after host-side verification;
the actual bytes it reads are not atomically bound to the checked digest under
concurrent writes. Creating a fresh snapshot for worker relaunch avoids reusing
an older published directory, but does not eliminate that pre-existing loader
window. CNOT, trained models, other GPU architectures, and any performance
improvement outside this synthetic fixture remain unverified here.
