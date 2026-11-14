# Session model snapshot reuse and lifetime

Status: experimental default-OFF Python-owned session cache implemented;
hardware correctness and paired performance remain pending. Rust still does not
own model preparation. This stage does not change inference precision or GPU
search kernels.

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

`model-snapshot-contender-green.log` records **89 passed** targeted CPU tests,
including real thread barriers, existing dispatch/reflection cases, model
identity and session lifetime. The deliberately failing test-first logs include
`model-snapshot-binding-red2.log`, `model-snapshot-helper-race-red.log`, and
`model-snapshot-contender-red.log`. The initial `model-snapshot-red.log` could
not create its missing temporary parent and is not implementation evidence;
`model-snapshot-red2.log` is the subsequent test-first run. A broader local
collection in `model-snapshot-cpu-final.log` failed because this Windows Python
environment's installed CayleyPy lacks `ModelConfig`; that suite was not claimed
as passed locally. The full suite is planned on the compatible remote image.

The existing native loader reopens mutable files after host-side verification;
the actual bytes it reads are not atomically bound to the checked digest under
concurrent writes. Creating a fresh snapshot for worker relaunch avoids reusing
an older published directory, but does not eliminate that pre-existing loader
window. Hardware correctness, CNOT, trained models, other GPU architectures,
and any performance improvement are not established by the CPU checks.
