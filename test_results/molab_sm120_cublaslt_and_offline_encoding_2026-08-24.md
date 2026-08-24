# Molab SM120 inference encoding and GEMM gate (2026-08-24)

## Scope

- Cube4 piece Transformer, `output_dim=24`, SM120 RTX PRO 6000 Blackwell.
- Original FP32 checkpoint is the quality truth; current FP16 is the accepted
  quality control.
- Weights for any narrow candidate are encoded offline. Runtime weight
  quantization is not accepted.

## Quality results on 20,480 real frontier states

| Candidate | top-1 vs FP32 | top-8 overlap vs FP32 | RMSE vs FP32 | Decision |
|---|---:|---:|---:|---|
| Current FP16 | 0.995605 | 0.992273 | 0.003960 | control |
| One-term native-scope block FP8 | about 0.954 | about 0.958 | 0.110914 | reject |
| Three-term residual decomposition | 0.995166 | 0.992029 | 0.004702 | quality pass |
| Four-term decomposition | 0.995166 | 0.992334 | 0.004544 | no material gain |
| MXFP8 E4M3 plus E8M0/32 | 0.959570 | 0.965190 | 0.083230 | reject |
| SmoothQuant alpha 0.5 | 0.968750 | 0.968840 | 0.076340 | reject |

Learned equalization, QAT distillation, single-operator one-term substitution,
and dense outlier-channel correction also failed the FP16-like ranking gate.

## Exact-shape native GEMM performance

Rows were 21,888, matching one production Transformer microbatch. Times are
CUDA-event means on Molab.

| K x N | old CUTLASS FP16 ms | workspace-free cuBLASLt FP16 ms | speedup | one-term block FP8 ms | three-term FP8 ms |
|---|---:|---:|---:|---:|---:|
| 256 x 768 | 0.046686 | 0.036989 | 1.262x | 0.030818 | 0.057439 |
| 256 x 1024 | 0.059605 | 0.047249 | 1.262x | 0.036932 | 0.071204 |
| 256 x 256 | 0.018608 | 0.014448 | 1.288x | 0.018503 | 0.031123 |
| 1024 x 256 | 0.063431 | 0.035197 | 1.802x | 0.049233 | 0.102117 |

The quality-preserving three-term packing is slower than exact FP16. The most
promising production direction is therefore modern exact FP16 GEMM first,
then a future narrow path only if sparse correction can pass both quality and
latency gates.

## Native graph integration

- Replaced the unsupported CUTLASS `Sm120` FP16 CollectiveBuilder experiment;
  CUTLASS reports that the SM120 builder supports only F8/F6/F4 MMA.
- Added an opt-in, zero-workspace cuBLASLt FP16 residual path.
- CUDA graph smoke completed five depths at `beam=2**16` with two Stream1
  lanes and no CUDA error.
- The correct production batching contract was re-established:
  `BEAM_B_MICRO=3584`, `BEAM_STREAM1_TRANSFORMER_MICRO=896`. The former is the
  pipeline transaction; the latter is the internal Transformer inference
  microbatch. With 24 accumulation slots, a full `2**25` frontier targets about
  391 Stream3 jobs.

## Incomplete terminal gate

The corrected `2**25`, depths `0..8` run reached depth 6 after the configuration
was validated, then the Molab sandbox endpoint returned HTTP 410. The sandbox
could no longer be queried, so no depth-8 timing or promotion claim is made.
The last incorrect `B_MICRO=896` run was explicitly terminated before the
correct profile was started.

Commits: `a2d673ff`, `f9d8393e`, `09c3c9b4`, `f4908750`, `bf9998c4`,
`1b360595` on `codex/stream1-generic-seq-align-impl`.

## Profile-native execution follow-up (implementation pending Molab gate)

- SM120 profile schema v3 now records `fp16_gemm_backend`, `target_sm`, and
  zero-workspace requirements alongside the per-operator offline encoding.
- Selection rejects native latency rows without that exact execution contract.
- Runtime SHA-verifies `profile.json`, derives any low-precision operator mask
  from the immutable profile, and propagates the selected FP16 backend into the
  Transformer network view. The prior manual
  `BEAM_STREAM1_TRANSFORMER_SM120_CUBLASLT=1` switch is no longer used.
- Exact cuBLASLt execution now covers QKV, FF1, attention-out, and FF2 GEMMs;
  bias and activation remain separate exact kernels for graph compatibility.
- Plan caches are thread-local, preserving two-lane launch concurrency.

This follow-up has not been compiled or benchmarked locally, by explicit user
requirement. Required Molab acceptance remains: compile, profile hash negative
test, CUDA graph correctness, FP32/FP16 ranking equivalence, and the identical
`2**25` depth-8 A/B against `80.2952 s` with about 391 Stream3 jobs.

### Next mixed candidate: activation-weighted low-rank error correction

The new research candidate computes block-FP8 bytes from FP32 weights and an
offline factorization of `D * (W_fp32 - W_fp8)`, where diagonal `D` is the
per-input-channel RMS measured on the real calibration frontier. Runtime math
would be `FP8(A,Wq) + FP16(A,L) * R`; ranks 4, 8, 16, and 32 are screened.
This targets roughly 2%-16% extra arithmetic depending on rank/shape instead
of the roughly 2x-3x work of the rejected three-term dense decomposition.
It is deliberately non-promotable until Molab proves ranking/frontier quality
and a native correction kernel proves net latency improvement.

### Comparable native timing artifact

`tools/sm120_native_benchmark_artifact.py` converts completed combined runner
logs into the exact `native_benchmarks.json` contract consumed by selection.
It rejects mismatched hardware, output dimension, beam, both microbatch levels,
inference concurrency, Stream3 job count, per-rank frontier size, missing depth
8, duplicate depth 8, and any fatal/OOM/overflow marker. This prevents a fast
microbenchmark or a differently configured solve from promoting a profile.
