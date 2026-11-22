# Rust host-history and process services — 2026-11-07

This is the host-services part of stage 3, following published `3807013a`.
Full profile-selection and CNOT control transfer remain pending; this report does
not close the entire stage or enable any CUDA optimization/public default.

All compilation and execution ran on the same owned 2xRTX3060 instance described
in STAGE2.md. Raw outputs, source-archive identities and SHA-256 are under
`stage3-host-3060/`. New source/archive maintenance on the local machine used D.

## Accepted behavior

- Rust owns exact whole-layer history plans, RAM/disk/static-hybrid stores,
  reservation bounds, disk-failure fallback and parent remapping. The native
  dispatcher lends a ready pinned CandidateMeta batch for one callback; no second
  CandidateMeta array is copied. The GPU-source event and host-reader completion
  are separate reuse conditions.
- The C ABI optional history extension retains the Rust callback owner through
  context destruction and the blocking CLI entry. Rust panic is contained at each
  callback. Native history transport was extracted into a separate header.
- Rust ProbeSession/ProbeCohort own retained calibration child processes, bounded
  JSON/log transport and group cleanup. PyO3 adapters release the GIL while waiting.
  Public calibration policy still uses its existing path until final switching.

## Results

| Check | Observed result |
| --- | --- |
| Rust workspace CPU contracts and process tests | passed; raw per-target counts in final-rust.txt |
| Actual CPU C ABI, including coarse CLI history and callback lifetime | passed |
| PyO3 resident/session/cohort adapters | passed |
| Native CTest, CUDA and LibTorch builds | 57 passed, 1 optional Transformer fixture skipped |
| Python regression suite | 510 passed, 25 skipped |
| RAM, disk, static-hybrid history: baseline vs Rust on one/two GPUs | paths, width and stable resident PIDs matched for solved/unsolved jobs |
| Same session comparison with the LibTorch-linked library | passed with `native_cuda_graph`; eager executor not selected in this run |
| Cancellation and failure of exactly one rank | all owned workers reaped |
| Ordinary Rust CLI and real two-rank torchrun | legal 3-move path and exact global width 64 |
| Static-hybrid session under memcheck, full leak checking | 0 errors; 0 bytes leaked in 0 allocations; exit 0 |

Retrospective scope correction: the earlier LibTorch-linked session test did not
set `BEAM_STREAM1_EXECUTOR=libtorch_eager`. It verifies that library's default
native executor. Raw evidence remains unchanged; STAGE3_SESSION.md records the
later explicit eager comparisons. The standalone LibTorch inference benchmark
in STAGE3_INFERENCE.md already uses its actual eager provider.

The sanitizer used `--report-api-errors no`. The retained unfiltered stage-2 log
contains NCCL kernel-query diagnostics explicitly handled by the matching upstream
source; see STAGE2.md for the exact scope and source link.

CPU tests include empty/out-of-order batches, complete-layer placement failure
despite sufficient combined capacity, required/available errors, integer overflow,
RAM pruning with route reconstruction, and a real write failure injected using a
read-only file descriptor. The fallback test verifies that earlier failed disk
layers cannot overwrite later planned RAM layers. Rust's stored-RAM counter reports
bytes remaining after pruning, rather than the reference's cumulative writes.

The first ordinary CLI test exposed missing width telemetry in Release output:
the path was valid but the parser had no effective-width value. The engine now
emits `GLOBAL_BEAM_WIDTH_EFFECTIVE` from its actual plan independently of debug
logging. The failed output is retained under `cli/`; the passing CLI/torchrun
run is under `cli-final/`. Search width and selection policy were not changed.

The exact-Hamming model is a deterministic integration fixture. Single-run times
are diagnostics; they do not meet paired performance or trained-model quality
acceptance. Host history currently serializes commits, and its per-job host arena
and pinned allocations are not yet retained across jobs. CUDA arenas, weights and
graphs remain resident. T4/Hopper/Blackwell acceptance and public switching remain
separate outstanding tasks.
