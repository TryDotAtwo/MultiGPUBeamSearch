# Molab SM120 sparse narrow-precision roofline

Hardware: Molab RTX PRO 6000 Blackwell Server Edition, SM120. CUDA tests were
run only in the visible foreground marimo notebook cell.

Workload: square GEMM `M=N=K=8192`, CUTLASS SM120 block-scaled structured
sparse tensor-op path, tile `128x128x256`, Stream-K scheduler. Throughput is
reported as dense-equivalent `2*M*N*K/time`; the physical 2:4 sparse operation
contains half as many non-zero multiply-adds.

| Encoding/path | Iterations | Kernel ms | Dense-equivalent TFLOP/s |
|---|---:|---:|---:|
| MXFP8 sparse, persistent | 100 | 1.177721 | 933.593 |
| MXFP8 sparse, Stream-K | 100 | 1.155442 | 951.594 |
| NVFP4 sparse, Stream-K | 100 | 0.567370 | 1937.909 |
| NVFP4 sparse repeat 0 | 200 | 0.567558 | 1937.269 |
| NVFP4 sparse repeat 1 | 200 | 0.567948 | 1935.935 |
| NVFP4 sparse repeat 2 | 200 | 0.567746 | 1936.627 |

Result: the requested 1.5 PFLOP/s gate is exceeded reproducibly. The three
200-iteration repeats span 1.935935--1.937269 PFLOP/s dense-equivalent.

This is a hardware roofline result, not yet a Cube4 model-quality result. The
next gate is offline FP32-vs-FP16-vs-NVFP4+2:4 calibration and real Transformer
shape benchmarking before this encoding can be selected for production.
