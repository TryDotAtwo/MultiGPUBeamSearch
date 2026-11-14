# Session model snapshot reuse and lifetime

Status: experimental default-OFF Python-owned session cache implemented. One
version has bounded two-RTX-3060 correctness and paired performance evidence;
the revised validation path requires a fresh hardware measurement. Rust still
does not own model preparation. This stage does not change inference precision
or GPU search kernels.

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
criteria on this fixture. The later one-pass snapshot check has not yet been
measured on hardware.

The first evidence archive is `.local/cache/model-snapshot-20261114/evidence-v1.tar.gz`,
SHA-256 `0935555814637bdebc474a59a14f602248a2026a94248affedc5361e24852d2e`.
All 408 content blobs and 1,308 manifest records were verified locally and four
samples restored; 14 rebuildable items were recorded as omitted. Compact
receipts and raw results are in `model-snapshot-3060/`.

The existing native loader reopens mutable files after host-side verification;
the actual bytes it reads are not atomically bound to the checked digest under
concurrent writes. Creating a fresh snapshot for worker relaunch avoids reusing
an older published directory, but does not eliminate that pre-existing loader
window. The revised version, CNOT, trained models, other GPU architectures,
and any performance improvement are not established by the CPU checks.
