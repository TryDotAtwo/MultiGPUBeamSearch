# Rust downstream calibration ownership

Scope: exact-capacity component/profile control, cached evidence, optional full
steps and cancellation. Public default switching, maximum/working-width control,
CNOT orchestration and CUDA bulk-operation changes remain separate work.

## Implementation

- Typed component receipts verify communicator ranks, exact admitted capacities,
  physical concurrent sort jobs, five complete timing samples and transport
  volumes. The original Python service-envelope arithmetic is retained as an
  independent oracle. Its output is a scheduling proxy, not a measured depth.
- Rust owns the bounded proposal/admission/measurement/selection loop and one
  native planner per cohort. Metadata and original probe rows remain inspectable.
  It does not copy Python's layout-byte memory prefilter: every candidate goes
  through fresh native admission. This finite shortlist is not a global optimum
  certificate and may explore different profiles from the Python controller.
- The scalar-MLP input-row budget is explicitly converted to native parent count.
  Frozen inference microbatch, exact logical width and numerical kernels stay
  unchanged. An unstable baseline has one recorded retry; rank/protocol failures
  are not reclassified as allocation failures.
- Component caches recompute their inputs and selection and readmit exact physical
  geometry. Full-step mode compares the baseline and at most two challengers,
  excludes one warmup and retains five whole steps per profile. Its threshold is
  5% separation, distinct from the existing inference-only 2% threshold and from
  paired CUDA-optimization promotion.
- Python submits one coarse descriptor through `DownstreamCalibration`;
  `rust_control.tune_downstream` retains model validation and hardware descriptors.
  Cancellation reaches rank initialization. Normal planner/pipeline cleanup does
  not cancel the parent job. Native rank errors retain bounded capacity diagnostics.

## Reproduced failures and corrections

- Scalar MLP initially confused input rows with parents. The failing CPU case
  and correction preserve `rows = parents * move_count` while keeping original
  candidate counts.
- A full-step profile at logical width 4096, flush 1024 and capacity 5120 overflowed
  Stream 4. The frozen `0eafb193` runner independently reproduces error 3002 with
  required/available details. Both logs are retained. This is an existing
  admitted-but-not-operational profile, not evidence of a Rust timing regression.
  The controller comparison explicitly uses flush/capacity 131072, sufficient for
  this fixture's whole gross expansion. The requested width is never reduced.
- The initial Rust rank error hid the native capacity text. A CPU reproduction
  and bounded log-tail propagation now preserve it in the coarse error.
- Real timing values drifted one ULP when read back through the default JSON
  parser, invalidating full-step content hashes. A bit-exact failing CPU test
  reproduces this. `serde_json` 1.0.154's `float_roundtrip` feature now preserves
  the original `f64` values. This affects host receipt parsing, not inference
  arithmetic. The pinned crate's feature source is archived with the results;
  see its [feature list](https://docs.rs/crate/serde_json/1.0.154/features).
- Component-only jobs do not require unused full-frontier fixture files. Full
  fixture mode still requires a complete rank cohort.

## Verification

Builds and tests run on the owned Vast 2xRTX3060 lease, not the local GPU.
Rust: 55 CPU tests passed. Python: 528 passed, 9 skipped. The complete native
and LibTorch eager suites passed after the final validation fix; binary hashes
and the compiled source manifest are retained.
The final raw receipt, source manifest, binary hashes and logs are in
`stage3-downstream-3060/`; `verification.json` records the counted CPU results.
The retrieved archive contains 1030 files / 9,841,989 bytes, SHA-256
`ea6749ac6007ae46a0d0bd18d17461b9854f3783ca14872c7e7dd50e235dda3d`.
Local per-file hashes were checked against the remote archive before publication.

The explicit GPU test covers native CUDA and actual `libtorch_eager`, on one and
two GPUs at exact logical width 4096: component/full first calls, cache hits,
forged full-cache rejection and fresh measurement, the actual public Python
model/options adapter, and cancellation during two-rank startup. Every executed
full-step rank log asserts the selected executor; there is no inference from the
library's linkage alone. Service-envelope values are compared with the original
Python oracle. The initial failed traces are preserved separately.

These are small deterministic Hamming-model/control fixtures. They establish
neither trained-model quality nor a production speedup. No CUDA optimization or
new precision mode is enabled. Working-capacity policy still needs an operational
gate; VRAM admission alone does not establish it. T4/Hopper/Blackwell hardware
acceptance and paired optimization measurements remain outstanding.
