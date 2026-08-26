# Molab SM120 dense NVFP4 FFN transport benchmark — 2026-08-26

## Contract

- GPU: Molab RTX PRO 6000, compute capability 12.0.
- Build: CUDA 13, CUTLASS block-scaled dense NVFP4, `sm_120a`.
- Transformer: Cube4, ReLU, output dim 24, `d_model=256`, `ff_dim=1024`.
- No beam-search architecture changes.
- The measured pipeline is `FF1 + bias + ReLU + NVFP4 output/scales -> FF2`.
  FF2 consumes the fused FF1 output and scale-factor buffer directly; no BF16
  `ff_hidden` is materialized between the two GEMMs.

## Results

| token rows M | iterations/run | runs | median pipeline ms | min..max ms | dense-equivalent TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 51,072 | 500 | 5 | 0.102888 | 0.102861..0.102975 | 520.5 |
| 204,288 | 200 | 5 | 0.413433 | 0.413210..0.413497 | 518.1 |

Control FF1 fused ReLU-to-NVFP4 at `M=51,072` remained green after a clean
`sm_120a` rebuild: `0.072258 ms`, `370.569 TFLOP/s`. The chained FFN pair was
`0.102763 ms` in the first 300-iteration run and `0.102888 ms` median across
five 500-iteration runs.

The traffic-only lower-bound probe copied 67 MiB device-to-device in
`0.057513 ms` (134 MiB physical read+write traffic). This is comparable to the
minimum traffic of a separate BF16-to-NVFP4 materialization pass, before its
amax reduction, scale generation, conversion, and launch overhead. Therefore
the fused path removes a material bandwidth-bound stage, rather than merely
renaming the intermediate buffer.

## Failure found and fixed during verification

A `sm_120` (non-accelerated target) rebuild compiled but failed at runtime with
`Arch conditional MMA instruction used without targeting appropriate compute capability`.
Reconfiguring to `BEAM_CUDA_ARCHITECTURES=120a` restored both the standalone
control and chained pipeline. Dense NVFP4 targets must be built for `sm_120a`.

## Status

- Standalone transport prototype: PASS.
- Direct FF1 output/scales reuse by FF2: PASS.
- Stable at 1x and 4x the production token-row shape: PASS.
- Production Stream1 integration: not enabled yet; accuracy/frontier-order
  comparison against FP32 and current FP16 remains required before opt-in.
