# Rust configuration, inference job and cache ownership — 2026-11-07

This increment follows published `8766a530`. The complete inference job now
runs in Rust: typed options, input preparation, artifact verification, cache
validation, probe selection and atomic publication. The thin package adapter
is directly tested; default public launch switching remains pending.

All builds/tests used owned Vast 59551108, two RTX3060 GPUs, USD0.1342222222/hour
including disk. Local source/artifacts are on D. The owned stop guard is bounded
at Unix1794077677; its maximum four-hour extension costs USD0.537 in compute/disk,
within the initial allocation. This is not a final invoice total.

`stage3-config-3060/manifest.json` hashes 630 files, 2,789,503 bytes.
Archive SHA-256:
`3fc2582c5d83d7e056cfac65b91b63e6ecd3ee2ba55ea456ae65f1a5e95c0fb2`.
`source-final.json` binds the final implementation tested on CPU; the final GPU
and signal fixtures are bound by `source-gpu-final.json`. Executed Rust binary
hashes are recorded separately. Later documentation edits are not executed code.

| Check | Observed result |
| --- | --- |
| Rust workspace | 44 CPU tests passed; one actual-FFI test explicitly ignored in this command and covered previously |
| Python regression suite | 527 passed, 9 skipped |
| PyO3 typed options and pre-cancellation | passed without launching a child |
| SIGINT through inference and search | KeyboardInterrupt propagated after owned worker cleanup, within four seconds |
| Complete inference jobs | native MLP, LibTorch MLP, ensemble on one/two physical GPUs passed |
| Cache reuse | no helper launched on a valid hit; raw receipts and recomputed estimates retained |
| Corrupt cache / changed model | invalid numerical receipt caused fresh measurements; changed model bytes rejected even before a cache hit |
| Thin public-package adapter | actual NativeOptions, graph/model contracts and hardware signature passed through one Rust call; second call hit the cache |
| Search-session regression | 24 native/actual-LibTorch-eager cases passed again after the interruptible PyO3 change, including two-GPU cancellation and rank failure |

`gpu-final/` contains the final request/cache and thin-adapter evidence.
`search-gpu/` contains the rerun of first/repeat/unsolved/model-change/beam-change/
collection cases. The Hamming fixture and helper numerical oracles establish
control compatibility, not trained-model quality or production speed.

Reproductions retained:

- `cleanup-red.log`: normal pool teardown incorrectly set its parent's cancel
  token, preventing later verified publication. Close now reaps its workers and
  marks the pool closed; an explicit cancellation remains independent.
- `signal-red.log`: main-thread SIGINT remained deferred while PyO3 blocked in
  Rust, eventually appearing during cleanup. Scoped Rust work plus 20 ms Python
  signal checks now cancels and joins the owned operation before propagation.
- Missing-interface RED logs cover configuration, artifact/cache APIs, whole-job
  preparation, PyO3 and the thin package adapter. An initial test syntax error
  and an incomplete source-sync attempt are retained separately and do not count
  as intended feature REDs; `cache-red-api.log` records the actual missing APIs.

The new cache policy uses 128 deterministic 16-move walks from the graph center.
This makes the measured input workload independent of the next search start.
The versioned sample policy and sample hash invalidate old cache identities.
All model/helper bytes, graph and device descriptors remain bound. Complete rank
timings, numerical success, reserve accounting, and unthrottled telemetry are
revalidated; claimed medians are recomputed. Capacity-limited profiles require
current selected-UUID free memory and are rejected after pressure lifts by more
than 32 MiB. A cache hit never replaces native memory admission.

No C++ source, CUDA arithmetic or precision policy changed. Downstream adaptive
profile policy/cache, full public-session/build integration, CNOT control,
CUDA bulk operations and hardware/performance acceptance remain outstanding.
