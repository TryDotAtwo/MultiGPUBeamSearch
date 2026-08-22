# Molab SM120 block-scaled FP8 primitive

## Scope

- Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0.
- Build/test location: Molab only.
- CUDA: 13.3.73; CUTLASS main `7107b05535f8977f5ecb9d01ee203205b1fd9bc4`.
- Branch: `codex/stream1-generic-seq-align-impl`.
- Feature: opt-in `BEAM_ENABLE_SM120_FP8`; production default remains unchanged.
- Integrated comparison baseline: Cube4 output_dim=24, beam `2**25`, exactly
  `depth_done=8`, 391 Stream3 jobs, 80.2952 seconds.

## TDD and correctness

The test-only RED commit failed in Molab because the implementation header did
not exist. After implementation, the SM120 test compiled for `sm_120a` and ran
on the physical GPU. An initial zero-output failure isolated an invalid device
conversion through `NumericConverterClamp`; native CUTLASS E4M3 construction
fixed the actual quantized bytes.

Final correctness output:

```text
stream1_transformer_sm120_fp8_cuda_tests=pass nmse=0.000415495 max_abs_error=0.0268555 workspace_bytes=0
```

The test also verifies exact zero preservation and scale-layout element counts.

## Real Cube4 shape benchmark

Each timing includes dynamic FP16-to-E4M3 activation quantization plus the
block-scaled GEMM. Weights are prequantized outside the timed loop. Five warmups
and 30 timed iterations were used with 51,072 rows.

| M | N | K | End-to-end ms | Effective TFLOP/s |
|---:|---:|---:|---:|---:|
| 51,072 | 768 | 256 | 0.110785 | 181.273 |
| 51,072 | 1,024 | 256 | 0.126435 | 211.780 |
| 51,072 | 256 | 256 | 0.082043 | 81.593 |
| 51,072 | 256 | 1,024 | 0.286033 | 93.613 |

Raw official CUTLASS GEMM-only probes were faster; the gap quantifies the
dynamic activation quantization cost, especially for K=1024. Therefore the
next optimization must fuse or amortize quantization and use the accuracy tuner
before production integration. No integrated speedup claim is made yet.
