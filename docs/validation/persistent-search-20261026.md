# Persistent native search

The opt-in NativeSession retains one local native process per GPU, uploaded
models, static device storage, streams and CUDA Graphs across compatible calls.
The resident production loop uses the existing per-task reset and recreates
search history. Start state changes reuse workers. Changed graph/model/binary,
selected devices, beam/depth or options invalidate the group before calibration.

CPU checks cover real subprocess mailboxes for two ranks, per-call log isolation,
worker failure and timeout cleanup, context ownership, model invalidation and
independent replay rejection. Existing backend/dispatch/preparation tests pass.

RTX 3060 12 GiB, real Cube555 Transformer + MLP FP16 blend (0.8/0.2), LibTorch:
- Two identical starts used PID 1244; both replayed [16,8,1].
- A changed start reused PID 1244 and replayed [11,0].
- Changing beam 4096 to 8192 rebuilt the group (PID 1264).
- All workers terminated on context exit.
- With autotuning enabled, the second call succeeded with both calibration
  functions replaced by functions that raise if called.
- A fresh session reused both validated inference and pipeline disk profiles.

One-GPU evidence archive SHA256:
0683a7ffac15c6367b85ebc53c514199302c775c90109e1b68769ee0ab2650b3
All 477 manifest members were hash-verified. No long solve or performance sweep
was run. Multi-GPU LibTorch acceptance is recorded below. Separate CUTLASS resident GPU acceptance is recorded below.

Working-frontier acceptance is recorded below for one concrete model/hardware
cohort. This does not certify128-GPU operation or a persistent service across
different physical machines.

Two RTX 3060 host: both resident and cold control failed before search at
CUB MergeSort during CUDA Graph capture (invalid device ordinal), on driver
535.161.07 and CUDA 12.8. This does not establish multi-GPU acceptance.
All 357 archived evidence members verified; archive SHA256
9c88db88241508ccaf83baae6329aa6aaf5733e33c399788946fcf868b316aff.
The owned test instance was deleted and its absence verified.

Kaggle 2xT4 found a controller-result contract bug in the first implementation:
all ranks completed, but only the reconstruction controller prints a solved
terminal record. Fixed in 7c38e2983f16a1c9a941b565e48365cafa3f6648.
The transport still requires every rank completion receipt. Solved results
require one controller record; unsolved results require one record per rank.
Targeted one-/two-rank controller contract checks: 2 passed.

## Completed Kaggle 2xT4 acceptance

Source 7c38e2983f16a1c9a941b565e48365cafa3f6648, private validation v4:
- Simple public-notebook flow completed from a clean GPU environment. Two calls
  retained PIDs908/909, returned independently replay-validated paths, and both
  processes were gone after explicit close.
- Real Cube555 Transformer+MLP FP16 blend0.8/0.2, LibTorch, beam4096/depth4:
  five calls retained PIDs756/757. Paths were [16,8,1], [16,8,1], [11,0],
  not-found, [16,8,1]. Every found path independently replayed to the center.
- Increasing beam to8192 created PIDs811/812 and returned [16,8,1]. Both new
  processes were gone on context exit. No stale stop flag/history after not-found.
- Solved calls had one controller terminal record; not-found had two rank records.
  Every request required matching completion receipts from both ranks.
- GPU tests were short correctness/lifetime checks, not a performance campaign.

Both Kaggle validation kernels reached COMPLETE. The first two setup attempts
used a CPU Docker pin and failed preflight; the GPU image pin is now corrected.
The previous full Cube555 notebook160 solve remains separate evidence; it was
not repeated for this change. Compatible profile disk reuse and forbidden warm
recalibration were separately verified on RTX3060 as described above.

## Follow-up validation boundaries

The working-frontier planner fix in `0b32691ae81a9ad8cc8bfe98b3ecf9d939fd272c`
compiled and ran on two Tesla T4 GPUs with the real Cube555 FP16
Transformer+MLP blend. Native admission selected 55,173,120 parents in 2.46s.
This is a memory-admission result, not a completed full-step result or a verified
efficient maximum. The subsequent six-depth-per-profile verification was
cancelled because the measured inference throughput projected hours of work.

The derived physical shard slab now reserves the configured flush and trigger,
so decreasing a logical frontier does not invalidate a frozen flush geometry.
Explicit manually supplied capacities remain strict. Logical top-k is unchanged.
The native regression for fixed-flush admission and memory monotonicity passed
on RTX3060 (runtime_arithmetic failures=0).

New full-frontier calibration checks its projected inference cost against the
time budget before starting full-frontier verification. Existing validated profile
cache hits bypass this new-work check. The projection is an estimate, not a
performance measurement or capacity guarantee. Twelve pipeline calibration
unit tests passed, including this cache ordering regression.

An earlier Kaggle request for L4 received two T4 GPUs and correctly failed
SM80+ preflight. Successful CUTLASS acceptance subsequently used RTX3060 below.
Loader diagnostics preserve the failing exit code and missing-library error;
the targeted regression passed. CUDA packages were built on CPU-only Kaggle
workers; paid hosts only executed the supplied binaries.

## Completed CUTLASS 2xRTX3060 acceptance

Python source6dd8564, native sourcecae5b9d, CUDA12.8, driver580.178.04:
real Cube555 Transformer+MLP FP16 blend0.8/0.2 on two RTX3060 GPUs.
Five calls retained PIDs519/520: paths [16,8,1], [16,8,1], [11,0],
not-found, [16,8,1]. All found paths independently replayed. Beam8192
created PIDs561/562 and replayed [16,8,1]. All workers exited on close.
Native arithmetic regression reported zero failures.
Evidence archive:163 members, SHA256
b89c3f5f85778cac6d40c05cb9536f23e8d0eaac19e73e0c69474372fdf1fcd1.

Natural frontier calibration in8a3395b grows the ordinary native search
and measures six saturated steps (one warmup, five measured). An early solution,
finite graph, timeout, or incomplete cohort cannot certify full-frontier timing.
Python fixture generation is bypassed when the native build advertises this
protocol. Targeted CPU tests:24 passed; the subsequent Python guard change
passed all13 pipeline-autotune tests. Native build and GPU acceptance completed.

## Working frontier: completed natural-growth validation

Source4cf894b470138fbc736403379b44c9dcefef78ad; native source8a3395b;
two RTX3060 12GiB, actual pretrained Cube555 MLP through CUTLASS.
The inference batch remained5120 parents throughout the downstream comparison.

| Global frontier | Median full step | Measured steps after warmup |
| --- | ---: | ---: |
| 19,994,624 |33.6795s|5|
| 39,989,248 |68.3022s|5|

Each cohort grew through ordinary native search, filled the exact global beam,
and completed six saturated steps per rank (one warmup plus five measurements).
The larger width retained98.619% of the smaller width's throughput. It is the
largest admitted width tested with this inference reserve and profile domain,
not proof of a globally optimal maximum over all possible configurations.
Baseline beat the two-sort-lane challenger; bounded search stopped with
no statistically separated gain. Capacity admission took2.96s. Full calibration,
including inference refinement, profile comparison and half-width validation,
took1177.89s. This optional maximum certification is not a subsecond operation.
Compatible profiles are reused before expensive recalibration.

Isolated Stream1 measured665,115 parents/s; complete search measured585,475
parents/s, about12% lower. These are separate workloads; this is not a
same-state measurement attributing the difference to cache or pipeline overhead.
The short production call also completed; this test did not attempt a deep solve.

Raw summarized measurements: [working-frontier-20261026.json](working-frontier-20261026.json).
Final evidence archive:283 verified members, SHA256
90b47e050a6828e6c80bbf995a643959b111e40161c34014e46428fa82479e0b.
The owned Vast instance57835090 was deleted and absence verified.

## Published notebooks

The simple Kaggle notebook is v7; Cube555 blend is v163. Both pin source4cf894b,
contain no dataset/competition attachments, and their downloaded published cells
match the reviewed source. Publication used QUICK_SAVE, without repeating the
already passed2xT4 acceptance. Clean2xT4 runtime acceptance above applies to
7c38e298; later natural-growth validation applies to2xRTX3060, not a new2xT4 run.

