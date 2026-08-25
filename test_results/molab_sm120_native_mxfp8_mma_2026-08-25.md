# Molab native SM120 MXFP8 MMA proof — 2026-08-25

## Environment

- Notebook: `444cubCayleyPy (Copy)` (`nb_tefJUzXQJw2NEN2JB1cF6J`)
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Compute capability: 12.0
- VRAM: 97,887 MiB
- CUTLASS: `7107b05535f8977f5ecb9d01ee203205b1fd9bc4`
- Target: `compute_120a`, `sm_120a`

## Native contract

- A: `mx_float8_t<float_e4m3_t>`
- B: `mx_float8_t<float_e4m3_t>`
- Scale: OCP UE8M0, vector size 32 selected by the CUTLASS builder
- Operator class: `OpClassBlockScaledTensorOp`
- Accumulator: FP32
- Output: BF16 in the isolated official CUTLASS fixture
- Iterations: 200 after CUTLASS setup/warmup

## Results

| Operator | M | N | K | Average ms | TFLOP/s | Verification |
|---|---:|---:|---:|---:|---:|---|
| QKV | 51,072 | 768 | 256 | 0.0555414 | 361.574 | Passed |
| FF1 | 51,072 | 1,024 | 256 | 0.0719130 | 372.345 | Passed |
| Projection | 51,072 | 256 | 256 | 0.0247378 | 270.603 | Passed |
| FF2 | 51,072 | 256 | 1,024 | 0.0555483 | 482.039 | Passed |

An initial 50-iteration 256x256 run also passed at 0.0253843 ms and
263.710 TFLOP/s.

## Interpretation

This is a hardware-path proof, not an end-to-end model result. It establishes
that the SM120 card can execute the desired native block-scaled MMA and that
the prior assertion was only an architecture-target mismatch. The current
production implementation still uses software FP32 block scales at 128x128
granularity and does not consume this native format. Next work must add:

1. immutable offline E4M3 plus UE8M0/32 weight artifacts;
2. dynamic activation encoding into the CUTLASS interleaved scale layout;
3. separate activation-encoding and GEMM timing;
4. full Stream1 and depth-8 A/B against FP16 and the old blockwise FP8 path;
5. ranking/frontier quality measurement as an independent Pareto dimension.
