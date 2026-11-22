# Rust inference calibration control — 2026-11-07

This increment follows published `963c24ef`. Rust now owns independent inference
processes and the complete bounded microbatch search/refinement/selection loop.
Existing C++/CUDA numerical probes and specialized inference kernels are unchanged.
Public configuration/cache orchestration and default adapter switching, CNOT
control, CUDA bulk improvements and other GPU-family acceptance remain pending.

All compilation and execution used owned Vast instance 59551108, two physical
RTX3060 GPUs, USD 0.1342222222/hour including 80 GB disk. Local work and retrieved
artifacts are on D. This hourly rate is not a final billing total.

## Evidence

`stage3-inference-3060/manifest.json` hashes 144 downloaded files, 695,215 bytes.
Archive SHA-256:
`e361eac7b0cff172141ed9459ebfed735320bfdc5dcc2efbecdfa2dfda9c47e7`.
Final CPU and GPU source identities are recorded in `source-manifest.json` and
`gpu-final/source-manifest.json`; executed binary hashes are also retained.
Documentation edited after execution is not claimed as executed source.

| Check | Observed result |
| --- | --- |
| Rust workspace CPU tests | 36 passed; the separate real-FFI test remains explicitly ignored in this command and was run in stage 2/host-history |
| PyO3 existing resident/probe/statistics checks | passed |
| PyO3 inference reuse and concurrent cancellation | passed; retained rank PIDs and complete owned-process cleanup |
| Public Python regression suite | 526 passed, 9 skipped in 51.60 seconds |
| Native MLP, LibTorch MLP, ensemble | reference/Rust probes passed on one and two physical GPUs |
| Per-backend batch cases | 8, 32, 16 at 129 parents; seven timings each, including partial final batches |
| Retained ensemble processes | one start per rank across all three batches; one-shot MLP helpers are reaped each time |
| Complete Rust inference tuning | passed at 2,048 parents/rank; tested microbatches 20, 24, 28, 32 on one/two GPUs |
| Legacy cache receipt validation | accepted complete Rust-produced rank and telemetry evidence |

The exact Hamming MLP and two-head ensemble test control and numerical-oracle
compatibility. They are not trained-model quality or production-performance
acceptance. The observed microbatch winner 32 is specific to this small fixture.
No CUDA optimization or precision mode was promoted. No native source changed in
this increment; native CTest/sanitizer results remain those documented previously.

## Reproduced defects and limits

- `deadline-red.log`: a short first measurement poisoned a retained worker's
  deadline, making both ranks fail on a later request. Workers now use the phase
  lifetime while each measurement enforces its own deadline.
- `oom-classification-red.log`: an invalid quality receipt could be treated as a
  memory bound when unrelated log text mentioned CUDA OOM. Quality/identity errors
  now remain non-capacity failures. Startup allocation rejection still closes all
  newly prepared probes, lowers their workspace bound and allows a smaller retry.
- `telemetry-red.log`: serialized Rust telemetry omitted the established aggregate
  `throttled` field. Serialization derives it from observed rows; missing data and
  observed throttling cannot yield an accepted candidate.
- `policy-red-fixed.log`, `tuning-red.log` and `adapter-red.log` record the missing
  interfaces before implementation. The earlier policy test syntax error is
  retained separately rather than presented as the feature's intended RED.

Independent inference workers contain no NCCL collectives, so a failed rank can
restart independently. Search/planning cohorts retain their all-rank failure rule.
Rust owns process groups, logs, bounded transport, deadlines, cancellation,
coarse/refinement candidates, statistics and reserve accounting. It preserves
the prior 2% inference-only selection threshold; optimization default promotion
still requires the separate five-pair full-step 5% / memory gate. Candidate order
and bootstrap use a versioned SplitMix64 policy, not Python's RNG ABI.

The public package retains its reference orchestration until the final switching
stage. The new Rust/PyO3 controller is directly exercised by these tests.
