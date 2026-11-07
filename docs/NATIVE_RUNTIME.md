# Rust/native runtime

The approved migration is in progress. Existing public Python entry points are
switched only after their comparative acceptance stage. Rust now owns host history
for its resident workers and ordinary CLI, plus calibration process primitives.
The frontier controller is available for comparative validation; public launch
and CNOT orchestration are still being transferred. No
CUDA sorting optimization or precision policy is enabled here.

An opt-in `BEAM_EXPERIMENTAL_RADIX128=ON` CMake build exercises full-key CUB
radix sorting in Stream 3/4 and the logical union. It requires CUB 2.7 / CUDA
12.8 or newer. The normal Python build explicitly sets the option OFF. The
experiment was slower on its measured RTX 3060 full-step fixture and is not a
default; see `test_results/refactor_20261107/STAGE4_RADIX128.md`.

`BEAM_EXPERIMENTAL_INDEX_SORT=ON` additionally sorts Stream 4 indices while
retaining compacted metadata in place. The Stream 3 radix and Stream 4 index
options can be selected independently; either experimental radix path requires
CUB 2.7 / CUDA 12.8 or newer. Ordinary Python builds turn both options off. The
index path preserves the full
hash and deterministic representative, reuses the existing value-B arena for
the two index arrays and gathers to the distinct survivor arena. A matched
two-RTX3060 fixture showed a lower sampled peak and no median slowdown when
Stream 3 remained at the default. Profile-specific public selection is not yet
implemented; see `test_results/refactor_20261107/STAGE4_INDEX_SORT.md`.

`BEAM_EXPERIMENTAL_STREAM4_SCAN=ON` independently selects CUB DeviceScan for
block counts and BlockScan for stable compaction. Counts remain on the device;
slot scratch is preallocated from actual CUB queries and reused in stream order.
The normal Python build explicitly leaves it OFF pending the measured profile gate.

`BEAM_EXPERIMENTAL_OWNER_RADIX=ON` independently groups Stream 3 candidates by
owner using stable CUB radix sorting with the sentinel included in the key width.
It reuses preallocated Stream 3 partition planes and scratch on the same stream;
the single-GPU short path is unchanged. Automatic Python builds leave it OFF.

## Build

Rust workspace builds do not require CUDA. `cargo build --release --workspace`
produces `beam-runner` and the PyO3 `_beam_runtime` library. The dependency graph
is pinned by `Cargo.lock`; the tested toolchain is recorded with each result.
Set `CARGO_TARGET_DIR` to a remote build directory (or an ignored directory on D
for local source-maintenance environments). The approved task runs builds on Vast.

The existing CMake configuration builds `beam_native` and, when
`BEAM_ENABLE_LIBTORCH_STREAM1=ON`, `beam_native_libtorch`. It uses the same CUDA
sources, explicit state specialization, CUTLASS and NCCL as the reference runner.
The LibTorch C++ ABI flag applies to all internal engine/control translation units.

Point `BEAM_NATIVE_LIBRARY` at the resulting absolute shared-library path before
invoking `beam-runner` with the existing native arguments. Without that variable,
the CLI looks for `libbeam_native.so` alongside its executable. Diagnostic modes
are forwarded through the coarse C ABI; `--resident-protocol-info` needs no CUDA
library. Torchrun and Slurm can launch one CLI process per rank with their existing
rank/environment contract.

## Context and sessions

`beam_context_create` checks ABI version and per-process exclusivity. `prepare`
copies argv and starts asynchronous resource preparation. `wait` observes either
readiness, task completion, timeout or terminal failure. `submit` copies the
small host job descriptor before returning and requires the next sequence.
Compatible jobs execute on the resident engine thread with the same resources.
`status`, `cancel`, bounded error retrieval and `destroy` complete the interface.

Rust `ResidentWorkers` owns only its launched process groups, atomic job/completion
files, sequence, deadline, cancellation token and log merge. A rank exit ends the
whole group. A two-second cleanup grace period precedes process-group termination
when a worker cannot leave an external collective. Paths and serialized states
are the process boundary; a GPU address is never a message payload.

The current native engine still uses process environment, cwd and stdout. Run it
in the dedicated worker, not in a notebook/Python process that owns CUDA tensors.
Independent Python sessions use independent worker groups. A `Context` cannot be
shared across Rust threads, and its native library cannot unload before it.

## Verification

The ABI-1 optional history extension installs six host callbacks before context
preparation. The coarse CLI entry has an equivalent blocking extension. Rust
owns RAM, disk reservations, whole-layer placement, fallback and parent remapping;
native code owns CUDA copies and pinned slots. A callback receives a borrowed
aligned CandidateMeta slice only after D2H completes, and contains every Rust
panic. The slot cannot be reused until that call returns. No extra CandidateMeta
array is copied at the boundary. Host history currently serializes commits;
performance promotion requires the paired measurements below.

`ProbeSession` and `ProbeCohort` own persistent calibration child processes, JSON
requests/responses and bounded log readers. Rank failures and deadlines close the
entire owned cohort. PyO3 releases the GIL for blocking operations. These are
runtime components; the public calibration policy has not switched yet.

## Calibration ownership

`FrontierCalibration` / `rust_control.calibrate_frontier` submits one complete
Rust job for inference, exact native admission and downstream selection. Numeric
widths are never reduced; a capacity rejection can only select another already
measured, quality- and telemetry-verified inference batch. Runtime/rank errors
propagate. The same cancellation token and overall deadline cover all phases.

Symbolic `max` separately records the finite multiset/VRAM/explicit descriptor
ceiling, shard-profile memory search, exact-width inference refinement and full
steps. A changed batch or reserve triggers fresh admission (at most three rounds).
The working search preserves an explicit larger arrival-batch floor and measures
fresh half-width full steps under the same execution profile, requiring at least
80% throughput retention and a saturated microbatch on every rank. It performs
at most three comparisons; it does not claim an exhaustive efficient maximum.
The existing `calibration_frontier_max_states` is a fixture-generation budget;
it is not implicitly turned into an execution-width limit. The private coarse
descriptor has a separate explicit ceiling for bounded diagnostic requests.
This policy does not reuse older smaller-width cache entries. The bounded fresh
comparison is easier to audit but costs a new full-step cohort on a repeated max
request. Only the current natural-growth native protocol supports this max job;
fixed-width jobs can also receive explicit full-frontier fixture artifacts.

The Python public launcher is not switched merely by exposing this job. Its
default transition and build orchestration require their own comparative checks.

`inference_job::Request` combines typed public options, graph/model descriptors,
content-bound artifacts, device identity and the native probe protocol. Rust
prepares inputs, validates cache evidence, owns all probe/selection loops and
publishes the result atomically. `rust_control.tune_inference` is the thin
comparison adapter; the public default remains gated on the final switch.

The versioned cache key includes the graph, deterministic center-based samples,
helper, protocol and all model files. Cached estimates are recomputed from the
complete all-rank receipts and unthrottled telemetry. A capacity-limited entry
needs a current selected-UUID free-memory snapshot; more than 32 MiB of released
pressure invalidates it. Every execution still needs fresh native admission.
Model/helper hashes are checked even on a cache hit. Hashing streams through a
64 KiB host buffer. `sha2` 0.10.9 and its dependency tree are pinned in Cargo.lock.

`downstream_job::Request` owns exact-capacity component calibration and the optional
complete-step comparison. Typed component receipts retain all 128-bit-key probe
checks in the existing native implementation. Rust validates every rank, records
telemetry, preserves the raw rows and computes the original service-work proxy.
Scalar MLP input-row batches are explicitly distinct from native parent batches.

The versioned proposal policy nominates at most four coarse challengers and asks
the native allocator to admit each one; it does not duplicate Python's layout-byte
memory prefilter. Full-step mode measures the baseline and at most two nominated
challengers, with one warmup and five complete repeats on the requested width.
It uses 5% statistical separation; the existing inference-only threshold is 2%.
Neither policy substitutes for the paired CUDA-optimization promotion gate.

Component caches recompute selection from every original receipt and fresh
all-rank admission with identical geometry. Full-step caches also recompute the
whole measured comparison. Missing ranks, quality failures and throttling cannot
become capacity rejections. An unstable baseline has one recorded retry. Native
overflow diagnostics survive process failure; a VRAM admission is not proof of
operational capacity for a particular graph/frontier.

Coarse PyO3 inference/downstream jobs and search-session calls run on a scoped Rust thread.
The Python thread checks signals every 20 ms, forwards cancellation and joins
owned cleanup before propagating KeyboardInterrupt. This avoids deferring a
notebook interruption until a long search timeout. An extra thread is created
per coarse call, never per candidate or kernel launch.

`PlanningSession` owns one retained process per rank and the exact-capacity
protocol. It validates every rank, preserves the requested logical beam exactly,
freezes the model microbatch, and permits only the five scheduling overrides.
Component probes use prepare-all/run-all; a rejected preparation is cancelled on
all ranks before another request. Capacity search retreats only on an explicit
native capacity rejection. Transport errors and rank failures abort the cohort.
The capacity API requires all five scheduling fields to be frozen and never
labels admission as full-step verification.

`PipelineProbe` owns admission, release of planning allocations, one coarse native
job per rank, deadlines, complete-step log validation and GPU telemetry. Both
legal frontier fixtures and saturated natural search are supported. The first
step is warmup; each following sample is global parents divided by the slowest
rank's complete-step duration. Missing ranks, changed width, incomplete output
frontiers, missing telemetry and throttling cannot produce an accepted profile.
One rank's early failure reaps the entire group without waiting for another rank's
NCCL timeout. Cancellation is an atomic token independent of the Python mutex.

`tune_profiles` runs the bounded shortlist in Rust. The generic evaluator trait is
implemented by the native PipelineProbe; Python is only the JSON/config adapter.
Its deterministic 2,000-resample bootstrap uses the versioned SplitMix64 policy,
so its confidence interval endpoints need not be bit-identical to the Python
reference's RNG. Neither policy changes inference arithmetic. The public cache
and candidate-generation policy still require migration before default switching.

`InferencePool` owns one independent scorer probe per rank. The ensemble protocol
retains model/arena processes across batches; existing native MLP and LibTorch
MLP helpers remain one-shot executables. All ranks start before readiness is
awaited or work is sent. A failed inference process is reaped independently;
healthy ranks are retained because these probes do not run NCCL collectives.
Cancellation and phase timeout close all owned processes. A per-measurement
deadline never becomes the retained process's lifetime limit.

The Rust inference controller owns coarse candidates, one bounded refinement,
capacity-bootstrap and fresh shared-proposal verification, telemetry, slowest-rank
statistics and reserve calculation. Only explicit CUDA allocation failure allows
the baseline to retreat to smaller rows; an invalid quality/identity receipt
cannot be reclassified as OOM by unrelated log text. Missing/duplicate ranks,
nonfinite timings, numeric errors and throttling reject evidence. The existing
microbatch threshold remains 2%; it cannot promote a CUDA optimization, whose
paired full-step gate is separate. Candidate ordering/bootstrap use the versioned
SplitMix64 policy. Numerical inference kernels are unchanged.

PyO3 exposes this complete controller and releases the GIL while it runs. The
public package still uses its Python reference policy until the final adapter
switch. Persisted inference receipts retain the established row/telemetry format;
the legacy cache validator accepts the produced real one/two-GPU receipts.

`paired_promotion` is a separate optimization gate: identical model/input/beam/
precision/ranks, at least five warmed complete-step pairs, and a conservative
upper time-ratio bound of 0.95, or measured peak-memory reduction in every pair
with an upper slowdown bound of 1.03. Incomplete/unstable evidence keeps a variant
experimental. It does not promote a component timing or a compilation result.

The single-GPU finalizer now fills `final_global_keep_count`, as the multi-GPU
path already did. Previously natural-growth calibration saw zero saturated steps
despite a valid frontier. The selected states and inference policy are unchanged.

- `cargo test --workspace`: layout, complete-hash order, domains, bounds, repeated
  jobs, stable worker PIDs, timeouts, cancellation and one-rank failure.
- `native_abi_tests`: the actual C ABI with an independent CPU engine fixture,
  including borrowed-input lifetime, invalid sequence, error text and cancellation.
- `BEAM_TEST_LIBRARY=... cargo test -p beam-native-sys --test context -- --ignored`:
  actual Rust calls through the CPU shared-library fixture.
- `rust/beam-python/tests/adapter.py`: repeated Python calls through the built
  PyO3 module, using owned CPU worker processes and no CUDA dependency.
- `tests/native_session_gpu.py`: real one/two-GPU comparison with the previous
  runner, exact global width, solved/unsolved jobs, independent path replay and
  resource reuse. Its exact Hamming MLP is a lifetime/correctness fixture.
- `tests/native_session_failure_gpu.py`: active cancellation and a missing native
  library in exactly one rank; all owned workers must be reaped.
- `tests/native_cli_gpu.py`: ordinary Rust CLI and two-rank torchrun with Rust
  history, exact global width and independent solution replay.
- `tests/native_planning_gpu.py`: reference/Rust exact plans, retained PIDs,
  repeated components and bounded capacity search on one/two GPUs.
- `tests/native_pipeline_gpu.py`: reference/Rust complete steps, explicit legal
  frontier and natural growth, followed by the Rust-owned shortlist loop.
- `tests/native_inference_gpu.py`: native MLP, LibTorch MLP and ensemble probes on
  one/two GPUs, all seven timing samples and partial final batches, retained PIDs,
  independent numerical oracles and the Rust-owned inference search loop.
- `tests/native_search_owner_gpu.py`: typed Rust session/result ownership, actual
  executor assertions, repeated jobs, model/beam changes, solution collections,
  cancellation and rank failure on one/two GPUs. `native_session_gpu.py --executor
  libtorch_eager` independently compares the eager executor with the baseline.
- Rust `SearchSession` validates complete permutation/collection paths, exact
  width and all-rank completion, owns compatibility/restart decisions and writes
  outcomes atomically. One deadline covers protocol preparation and execution;
  a separate parent cancellation token survives normal worker replacement.
- `rust/beam-python/tests/inference.py`: retained rank PIDs and active cancellation
  from another Python thread. Rust tests also reproduce deadline poisoning and
  distinguish startup OOM from an invalid quality receipt.
- Rust history tests cover disk-first reservations, out-of-order ready batches,
  empty batches, capacity diagnostics, RAM pruning and injected disk-write failure
  without overwriting future RAM reservations. Probe tests cover retained PIDs,
  deadlines, line parsing and cleanup when one rank exits.

The private extension can be packaged with `maturin build --release --manifest-path
rust/beam-python/Cargo.toml --out <wheel-directory>`. This component wheel does not
replace the public Python package. The recorded build used Rust 1.99.1, Python
3.11 and the CPython 3.10+ stable ABI; an older minimum Rust toolchain has not been
accepted.

These checks do not establish trained-model quality or a performance improvement.
Performance promotion requires the paired full-step measurements in the approved
plan. Per-device compilation and physical acceptance remain separate records.
