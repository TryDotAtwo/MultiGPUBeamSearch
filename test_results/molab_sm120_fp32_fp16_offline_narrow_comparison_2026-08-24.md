# Molab SM120 FP32 / FP16 / offline narrow-weight comparison

## Contract

- Hardware: RTX PRO 6000 Blackwell Server Edition, `sm_120`.
- Workload: Cube4 piece Transformer, `output_dim=24`.
- Quality corpus: 20,480 real frontier states, depths 4 through 8 (4,096 per depth).
- Truth: original FP32-exported checkpoint weights.
- Control: current FP16 runtime export.
- Candidate weights are encoded offline from the FP32 source. Runtime/startup weight quantization is not allowed.

## FP16 control versus FP32 truth

| metric | FP16 |
|---|---:|
| top-1 agreement | 0.99560547 |
| top-8 set overlap | 0.99227295 |
| threshold-band agreement | 0.99327827 |
| global overlap | 0.99824219 |
| pair inversion rate | 0.01057944 |
| logit RMSE | 0.00395994 |

## Offline E4M3 weights with exact FP16 activations

This is a quality-only diagnostic. It used FP16 matmul with the weights rounded
offline from FP32, because SM120 narrow Tensor Core MMA does not provide an
E4M3 x FP16/BF16 input path. CUTLASS example 87b uses E4M3 for both A and B;
BF16 is its C/D output type.

Best single operator by top-8 overlap was `blocks.3.ff.3.weight`:

| metric | candidate vs FP32 | candidate vs FP16 |
|---|---:|---:|
| top-1 agreement | 0.99277344 | 0.99296875 |
| top-8 set overlap | 0.99063110 | 0.99198608 |
| threshold-band agreement | 0.98985328 | - |
| global overlap | 0.99589844 | - |
| pair inversion rate | 0.01268489 | - |
| logit RMSE | 0.00941571 | - |

Aggregate candidates degraded further: all-FF top-8 `0.97588501`, all-attention
`0.97845459`, all-core `0.96810303`. The ordered cumulative series degraded
monotonically through nine operators, so it was stopped after all 16 singles,
all three aggregate families, and nine cumulative points (28 observations).

Persistent Molab artifact:
`/marimo/storage/cayleypy-cube4/sm120_quant_calibration/weight_only_fp8_fp32source_probe_detached/metrics.json`.

## Native full-FP8 throughput

The existing native kernel keeps weights pre-encoded and quantizes activations
per invocation. At the production Transformer microbatch (`384 * 57 = 21,888`
rows), direct CUDA-event measurements were:

| GEMM MxKxN | FP8 end-to-end ms | FP16 ms | speedup |
|---|---:|---:|---:|
| 21888x256x768 | 0.030797 | 0.047204 | 1.533x |
| 21888x256x1024 | 0.036885 | 0.060858 | 1.650x |
| 21888x256x256 | 0.018573 | 0.019124 | 1.030x |
| 21888x1024x256 | 0.049287 | 0.063716 | 1.293x |

These are kernel-level opportunities, not accepted model speedups. Full-FP8
quality and the previously reconstructed frontier diverge too strongly.

## Output correction probe

A tiny offline-fitted `24x24 + bias` correction was tested after an all-core
FP8 model, using disjoint even/odd calibration and holdout halves. It improved
neither enough nor stably: the FP32-target holdout top-8 overlap remained about
`0.9627`, far below the FP16 control `0.9923`. This demonstrates that the main
error is state-dependent and accumulated inside the Transformer, not a simple
output-head affine distortion.

Persistent Molab artifact:
`/marimo/storage/cayleypy-cube4/sm120_quant_calibration/fp8_output_correction_fp32source/metrics.json`.

## Decision

No tested narrow candidate is eligible for an immutable production profile.
The selector must remain fail-closed. The next performance work should preserve
FP16 numerical behavior and target fusion/scheduling, or introduce a selective
FP16 rescue mechanism with a separately approved cross-stream design and an
exact reconstructed-frontier gate.
