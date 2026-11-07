# Rust exact planning and full-step control — 2026-11-07

This increment follows published `531f0435`. It implements retained native
planning, bounded capacity search, complete-step process/telemetry ownership,
shortlist selection and the paired optimization acceptance calculation in Rust.
Public inference calibration, candidate-generation/cache policy, CNOT orchestration,
CUDA bulk optimizations and final public switching remain outstanding.

All builds/tests ran on owned Vast instance 59551108, 2xRTX3060, at
USD 0.1342222222/hour including its 80 GB disk. No local compilation was used.
Source changes and downloaded evidence are on D. This is not a final billing total.

## Evidence

`stage3-planning-3060/manifest.json` hashes 193 downloaded files, 866,528 bytes.
The archive SHA-256 is
`ea40072c8feb17564c2119cd53df7e99e0090fd0362f05604b3aad1da96cafab`.
The exact uploaded source file hashes are in `source-manifest.json`. Documentation
written after execution is not represented as executed source.

| Check | Observed result |
| --- | --- |
| Rust workspace CPU tests | 28 passed; real shared-library FFI test remains explicitly ignored in this command and was executed in the preceding stage |
| PyO3 retained workers/probes, cancellation from another Python thread, statistics | passed |
| CUDA/LibTorch rebuild and CTest | 59 passed, 1 optional Transformer fixture skipped |
| Single-GPU finalizer, partial/full frontier | passed after the reproduced missing-counter bug was fixed |
| Two-rank final union, staged arrivals, complete pipeline and empty sending rank | passed in all four modes |
| Reference/Rust plans on one/two GPUs | exact widths 5, 64 and 65; identical layout fields; retained process IDs |
| Repeated prepared component probes | passed on one/two GPUs |
| Rust bounded capacity search | reached the supplied ceiling 68 at alignment 4; records admission only, no capacity maximum or performance claim |
| Reference/Rust full steps on one/two GPUs | both explicit legal frontiers and natural growth passed; warmup plus five measured depths, exact global width 64 |
| Rust-owned admit/release/measure/select loop | baseline and outer-double profiles completed on one/two GPUs |

The native planning budget snapshot can differ by CUDA context bookkeeping;
the comparison checks identical exact layouts and capacities and validates each
rank against its own observed budget. GPU telemetry contains all selected devices.
The deterministic Hamming scorer is an integration fixture, not a trained-model
quality or production throughput acceptance test. No CUDA optimization was
enabled by these measurements, and no new precision mode was introduced.

## Reproduced defect and fix

The initial natural-growth run solved its 20-move fixture but reported zero
saturated steps. The single-GPU finalizer returned correct selected states while
leaving `final_global_keep_count` at zero. The independent final-union fixture
reported `global_hits=1,1,0,0` with `dispatcher_union=FAIL final_global_keep=0`.
That RED output is retained in `single-count-red.txt`.

The single-GPU path now sets its global count to the actual selected frontier
count. Selection, ties, hashes and inference arithmetic are unchanged. New partial
and full frontier CTests pass; two-rank behavior and natural growth also pass.
The earlier failed Python-reference run is retained under `pipeline-gpu/`;
the completed comparison is under `pipeline-gpu-final/`.

## Statistical and lifecycle limits

The Rust bootstrap uses a versioned deterministic SplitMix64 generator, so exact
confidence interval endpoints can differ from Python's RNG. The optimization
promotion API separately requires at least five matched complete-step pairs and
uses the conservative upper time-ratio bound (0.95, or 1.03 with measured memory
reduction in every pair). It rejects changed workload identity, missing ranks,
failed correctness/admission, throttling and unstable timing evidence.

CPU process tests cover retained PIDs, out-of-order rank failure, timeout, active
cancellation and draining more JSON log records than the bounded queue capacity.
Typed planning tests cover exact quotas at u64 limits, duplicate/missing ranks,
capacity required/available diagnostics, frozen microbatch and cancelling a
partly prepared cohort. These do not replace separate full hardware acceptance
for T4, Hopper and Blackwell or the final whole-plan review.
