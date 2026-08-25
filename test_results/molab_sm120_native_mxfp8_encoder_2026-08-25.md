# Molab SM120 native MXFP8 encoder benchmark — 2026-08-25

## Scope

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0.
- CUDA: 13.0 toolchain supplied by Molab.
- CUTLASS: `7107b05535f8977f5ecb9d01ee203205b1fd9bc4`.
- Solver commit: `77422d42`.
- Native instruction contract: `OpClassBlockScaledTensorOp`, E4M3 A/B,
  UE8M0 scale factors, scale-vector size 32, `sm_120a`.
- Work was compiled and executed on Molab. No local CUDA execution was used.

## Correctness gate

The benchmark encodes deterministic FP16 activations and immutable-style
`N x K` row-major weights, executes native MXFP8 GEMM, and compares the FP16
result against an FP32 CPU matmul oracle for a `128 x 128 x 128` problem.

| metric | value |
|---|---:|
| RMSE | 0.015659381 |
| relative RMSE | 0.032576885 |
| maximum absolute error | 0.065142781 |
| per-row top-1 agreement | 0.984375 |

This is a primitive/layout gate, not the final model-quality gate. Production
promotion still requires FP32-versus-FP16-versus-MXFP8 evaluation on real
Cube4 frontiers and a full depth-8 search checksum/quality comparison.

## Throughput at production inner microbatch

`M=51,072`; 10 warmups and 100 measured iterations. Weight encoding is outside
the timed path. Activation encoding, GEMM, and their combined latency are timed
separately.

| operator | N | K | activation encode ms | GEMM ms | end-to-end ms | GEMM TFLOP/s | end-to-end TFLOP/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| QKV | 768 | 256 | 0.059437 | 0.043136 | 0.102487 | 465.562 | 195.949 |
| FF1 | 1024 | 256 | 0.059471 | 0.067290 | 0.122782 | 397.928 | 218.082 |
| projection | 256 | 256 | 0.059426 | 0.018513 | 0.077932 | 361.590 | 85.897 |
| FF2 | 256 | 1024 | 0.227662 | 0.048878 | 0.278737 | 547.825 | 96.063 |

## Decision

The hardware-native path is real and fast enough to proceed. Runtime activation
encoding is now the dominant cost for projection and FF2, so production work
must keep weights pre-encoded and optimize/fuse activation packing. The backend
is not enabled in the solver yet; fail-closed FP16 remains the production path
until real-frontier and depth-8 gates pass.
