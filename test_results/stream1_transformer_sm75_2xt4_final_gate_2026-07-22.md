# Stream1 transformer SM75 / 2xT4 final gate

Date: 2026-07-22
Source commit used by Kaggle: `fe3cb95`
Reproducible ref: `stream1-transformer-sm75-gemm-v21-fe3cb95`

## Correctness contract

- Every accepted timing used a complete binary score dump, not the printed checksum/digest.
- The v21 broad sweep produced 210/210 exact dumps.
- The v22 focused gate produced 280/280 exact dumps and no runtime errors.
- All v22 dumps matched SHA-256 `6524c19ff92c7c87263eb9c1e3d2d64ccd20c6ca117e803e6b44770f226644ef`.
- The v24 balanced shape/policy gate produced 200/200 exact dumps and no runtime errors. Hashes were stable within each GPU/shape (`6524c19f...644ef` for GPU0/b384 and `e29fe6d...398d` for GPU1/b256).
- The final local SM86 regression passed CTest 18/18 and reproduced the canonical full-score SHA-256 `a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94` on an idle RTX 3070.

## Accepted SM75 policy

The accepted T4 policy at `b_micro=384`, concurrency 1 is:

- QKV: CUTLASS `m128n128`, identity swizzle 8, default two-stage SM75 mainloop.
- FF1: CUTLASS `m128n128`, warp `64x32`, stages 2, identity swizzle 1.
- Attention output and FF2: CUTLASS `m128n128` with the byte-exact fused residual epilogue, stages 2, identity swizzle 2.
- Final projection: CLS-only optimization enabled while attention itself remains the full accepted FMHA path.

The QKV stages experiment established that CUTLASS already instantiates the selected SM75 QKV kernel with a two-stage mainloop. An explicit three-stage SM75 instantiation does not compile, so the policy contract now reports SM75 QKV stages 2 and rejects stages 3. This is a contract correction, not a new hot-path kernel. The schema-v3 hardware/workload cache now also accepts the measured FF1 `m128n128w64n32` winner, so the exact SM75 bundle can be stored and resolved without a manual-only preset; signature mismatches still fail closed.

## Performance gates

### Isolated transformer, v22

| GPU | CLS baseline | selected | speedup | paired wins |
|---:|---:|---:|---:|---:|
| 0 | 753,767.9 cand/s | 1,405,327.9 cand/s | +86.44% | 20/20 |
| 1 | 758,551.3 cand/s | 1,399,623.0 cand/s | +84.51% | 20/20 |

Aggregate selected throughput was 2,804,950.8 candidates/s versus 1,512,319.2 candidates/s for the exact CLS baseline.

### Stream1 -> Stream2 -> Stream3 integration, pipeline v4

| GPU | baseline | selected | speedup | paired wins |
|---:|---:|---:|---:|---:|
| 0 | 751,330.0 cand/s | 1,361,530.0 cand/s | +81.22% | 20/20 |
| 1 | 748,369.5 cand/s | 1,380,430.0 cand/s | +84.46% | 20/20 |

The large isolated gain therefore survives the real pipeline gate and is accepted.

## Rejected follow-ups

- Shape: balanced v24 measurements retained `b_micro=384`, concurrency 1. Aggregate b384 median was 2,589,786.8 candidates/s; b256 was 2,559,444.2 and won only 10/40 paired comparisons.
- Attention: q1/q32 plus exact32 looked promising in the isolated v24 gate, but pipeline v5 improved only +0.53% on GPU0 and +2.21% on GPU1. Both are below the predeclared 3% gate, so the combination is rejected.
- The 80 v23 shape errors were benchmark-harness omissions, not CUDA or math failures: requested b128/b192/b320/b448/b640 values were absent from the static sweep, so successful processes emitted no matching row. The sweep now includes those sizes and therefore fails observably rather than silently dropping requested shapes.

## Decision

Accept the SM75 GEMM bundle above and keep `b_micro=384`, concurrency 1. Do not enable the combined attention candidate. The measured SM75 contour is saturated for the tested CUTLASS tile/swizzle/stage, launch-shape, CLS, FMHA max-K, LayerNorm, and attention tiling axes under the 3% integrated acceptance rule. Further material improvement needs a new fused kernel/dataflow or lower-level profiling on hardware that exposes such a bottleneck; it is not justified by another small policy permutation.

## Evidence

- `test_results/kaggle_stream1_transformer_2xt4_v21_sm75_2026-07-22/stream1_transformer_t4_summary.md`
- `test_results/kaggle_stream1_transformer_2xt4_v22_sm75_2026-07-22/stream1_transformer_t4_summary.md`
- `test_results/kaggle_stream1_transformer_2xt4_pipeline_v4_sm75_2026-07-22/stream_pipeline_gate_summary.md`
- `test_results/kaggle_stream1_transformer_2xt4_v23_sm75_2026-07-22/stream1_transformer_t4_summary.md`
- `test_results/kaggle_stream1_transformer_2xt4_v24_sm75_2026-07-22/stream1_transformer_t4_summary.md`
- `test_results/kaggle_stream1_transformer_2xt4_pipeline_v5_attention_2026-07-22/stream_pipeline_gate_summary.md`
- `.gpu_queue/logs/cb910dc53f9b.log`
