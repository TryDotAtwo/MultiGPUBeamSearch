# Molab dense NVFP4 Cube4 shape study

GPU: RTX PRO 6000 Blackwell Server Edition (SM120). All CUDA compilation and
measurements ran in visible foreground Molab notebook cells. No structured
sparsity is used in this report.

The Cube4 Transformer has `d_model=256`, `ff_dim=1024`, sequence length 57,
and a measured external microbatch of 896 parents. Linear layers therefore use
`M=896*57=51072`, while their fixed `N,K` dimensions remain narrow.

## Dense NVFP4, BF16 output

| Scheduler/tileK | Layer | M | N | K | ms | TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| Stream-K/256 | QKV | 51072 | 768 | 256 | 0.041030 | 489.459 |
| Stream-K/256 | FF1 | 51072 | 1024 | 256 | 0.073176 | 365.920 |
| Stream-K/256 | projection | 51072 | 256 | 256 | 0.018469 | 362.458 |
| Stream-K/256 | FF2 | 51072 | 256 | 1024 | 0.032898 | 813.917 |
| static/256 | QKV | 51072 | 768 | 256 | 0.036909 | 544.104 |
| static/256 | FF1 | 51072 | 1024 | 256 | 0.065803 | 406.920 |
| static/256 | projection | 51072 | 256 | 256 | 0.016422 | 407.625 |
| static/256 | FF2 | 51072 | 256 | 1024 | 0.030940 | 865.444 |
| Stream-K/128 | QKV | 51072 | 768 | 256 | 0.041321 | 486.008 |
| Stream-K/128 | FF1 | 51072 | 1024 | 256 | 0.072391 | 369.889 |
| Stream-K/128 | projection | 51072 | 256 | 256 | 0.018482 | 362.204 |
| Stream-K/128 | FF2 | 51072 | 256 | 1024 | 0.032784 | 816.745 |
| static/128 | QKV | 51072 | 768 | 256 | 0.036904 | 544.173 |
| static/128 | FF1 | 51072 | 1024 | 256 | 0.067403 | 397.260 |
| static/128 | projection | 51072 | 256 | 256 | 0.018246 | 366.879 |
| static/128 | FF2 | 51072 | 256 | 1024 | 0.030755 | 870.640 |

Best per layer: static/256 for QKV, FF1, projection; static/128 for FF2. The
four-linear weighted aggregate rises from about 485 TFLOP/s with Stream-K/256
to about 536 TFLOP/s, a 10.5% improvement. Square 8192^3 dense NVFP4 reaches
1430.991 TFLOP/s, proving the gap is shape-dependent rather than failure to
select the native SM120 FP4 path.

## Fused ReLU plus NVFP4 output

| Layer shape | ms | TFLOP/s |
|---|---:|---:|
| QKV | 0.055492 | 361.895 |
| FF1 | 0.072271 | 370.503 |
| projection | 0.022582 | 296.441 |
| FF2 | 0.038963 | 687.224 |

The fused epilogue is slower as an isolated GEMM. It remains a valid pipeline
candidate only where it removes a separate BF16 store, ReLU, and activation
quantization pass (principally FF1 to FF2). It must be judged end-to-end.

## Conclusions

- Increasing parent batch cannot widen K/N; M is already 51072.
- QKV is already fused. Cross-layer grouped GEMM is blocked by true attention
  and residual dependencies.
- Static scheduling is the measured default for these large-M narrow-K shapes.
- The practical bypass is fusion and keeping selected intermediates narrow,
  not synthetic batching. Residual and normalization boundaries remain
  BF16/FP32 unless an exact fused implementation replaces their traffic.
- Next benchmark: BF16 FF1 + standalone ReLU + NVFP4 quantization versus fused
  FF1 ReLU-to-NVFP4, followed by accuracy calibration.

Official references:

- https://docs.nvidia.com/cutlass/latest/
- https://docs.nvidia.com/cutlass/latest/CHANGELOG.html
- https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/006_block_scaled_gemm.html
