# Initial Rust/native integration — 2026-11-07

Baseline source: `0eafb19369262a7d254058a0b4839bea7dfa3334`.
Implementation follows consolidation `3a905f479beb5adaa3f089dd455dd12edeecdb03`.
All builds and tests below ran on the owned Vast instance 59551108: two RTX 3060
12 GiB, CUDA 12.8, driver 595.91.07, PyTorch 2.9.0, NCCL 2.27.5, Rust 1.99.1.
Local operations were source/archive maintenance on drive D.

## Evidence

Raw logs and per-file SHA-256 are in `stage2-3060/manifest.json`. The source archive
hash recorded by the native build binds the uploaded source used by that build.

| Check | Observed result |
| --- | --- |
| Baseline CTest, state 96/112, 24 moves, SM86 | 56 passed, 1 skipped |
| Native CTest, same geometry | 57 passed, 1 skipped |
| Python regression suite, CPU | 510 passed, 25 skipped |
| Rust workspace CPU tests | 8 passed; real-FFI test separately enabled and passed |
| Built PyO3 module and isolated wheel install/import | passed |
| Reference vs Rust/native session, one/two physical GPUs | 3 successive jobs per session, stable PIDs, same paths, exact global width 64 |
| Same session comparison using the LibTorch-linked library | passed with the default `native_cuda_graph` executor; not LibTorch eager acceptance |
| Active cancellation and failure of exactly one rank | entire owned process group reaped |
| Unfiltered Compute Sanitizer memcheck | 16 CUDA API errors from NCCL kernel-attribute queries; not a clean pass |
| Memcheck with `--report-api-errors no` | zero reported memory errors, exit 0; narrower evidence |

Retrospective scope correction: linking LibTorch does not select `libtorch_eager`.
Those earlier session runs omitted `BEAM_STREAM1_EXECUTOR`; their bootstrap logs
show the native default. Raw evidence and hashes are unchanged. Explicit eager
executor assertions and baseline comparisons are recorded in STAGE3_SESSION.md.

The sanitizer's unfiltered errors are `cudaErrorNoKernelImageForDevice` at
`ncclInitKernelsForDevice`/`cudaFuncGetAttributes`, in both the unchanged baseline
and the new library. The application fixtures still completed. The raw unfiltered
log is retained in `sanitizer/session-memcheck.txt`; disabling API reporting in the
second run does not close the unfiltered gate or validate all NCCL behavior.

Source follow-up: the matching upstream NCCL 2.27.5 implementation explicitly
continues past `cudaErrorNoKernelImageForDevice` while querying available kernels
([NVIDIA source, lines 37–39](https://github.com/NVIDIA/nccl/blob/v2.27.5-1/src/enqueue.cc#L37-L39)).
This explains the recorded API diagnostics; it does not change the raw sanitizer
result or turn the API-disabled run into full API-error coverage.

`stream1_transformer_cuda_tests` was skipped because its model fixture was absent.
The GPU session fixture uses an exact Hamming MLP, not a trained quality benchmark.
It covers solved and unsolved requests and independently replays each solved path.
Single-run elapsed times in its JSON are diagnostics, not performance promotion.

An unchanged-baseline Cube555 test assumed bit identity between FP32 `addmm` and
`mm + bias`; it failed on this CPU with maximum difference 1.49e-8. The test now
retains exact exported weight/bias checks and compares both computations against
an independent FP64 dot product with an FP32 forward-error bound. Inference
implementation and numerical policy were not changed.

## Scope and resource record

This stage adds the versioned C ABI, lazy native loading, Rust contracts, resident
rank supervision, CLI and private PyO3 module. Existing public launches remain the
reference until subsequent compatibility acceptance. History/profile ownership,
CNOT orchestration, CUDA sorting/compaction, per-architecture validation and paired
performance remain outstanding.

The earlier 2x3090 lease 59546637 was stopped, its results downloaded and hash
verified, then destroyed. Estimated compute/disk cost was approximately USD 0.34
before final traffic accounting. The replacement 2x3060 lease costs USD
0.134222/hour including 80 GB storage and has a six-hour stop guard. The total
authorized cap remains USD 50; other projects' leases were not modified.
