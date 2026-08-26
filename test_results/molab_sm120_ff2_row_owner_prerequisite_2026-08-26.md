# Molab SM120 FF2 row-owner prerequisite — 2026-08-26

## Objective

Test whether a generic dense NVFP4 CUTLASS FF2 kernel can give one CTA ownership
of the complete Cube4 `d_model=256` row. Full-row ownership is required for a
single-kernel `FF2 + bias + residual + LayerNorm + next-layer NVFP4/SF`
epilogue; an `N=128` CTA cannot compute exact row statistics without another
CTA or kernel.

## Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (`sm_120`).
- CUTLASS: current upstream checkout in Molab.
- CUDA compiler: 13.3, target `sm_120a`.
- Shape: `M=51,072`, `N=256`, `K=1,024`.
- Kernel tile: `128x256x128`, dense block-scaled NVFP4, FP32 accumulation.

## Result

The kernel compiled and its CUTLASS reference verification passed:

```text
Disposition: Passed
Problem Size: 51072x256x1024
Avg runtime: 0.227757 ms
GFLOPS: 117566
```

The result proves functional support for an `N=256` row-owner tile, but rejects
the generic tile as a performance candidate. It is far below the previously
measured fixed `N=128` FF2 throughput (about 870 TFLOP/s at the same production
row count). The generic `128x256x128` builder therefore must not be enabled in
Stream1.

## Next kernel contract

Do not return to grouped FF1. The next implementation is a fixed-shape SM120
CTA with two independent 128-column warp-level block-scaled MMA consumer
groups. Together they own one 256-column row tile. Their FP32 partial output
remains on-chip, then cooperatively performs bias, residual, exact row
sum/sumsq, LayerNorm, and next-layer NVFP4 plus UE4M3 scale emission. Start
with an M tile of 64, then 32/16 if register pressure or occupancy requires it.

The Molab sandbox expired before the `64x256x128` follow-up compile began. No
performance or correctness claim is made for that unexecuted variant.
