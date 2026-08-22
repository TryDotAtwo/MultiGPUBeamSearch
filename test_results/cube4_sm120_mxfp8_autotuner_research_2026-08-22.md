# Cube4 SM120 MXFP8 Autotuner Research — 2026-08-22

## Scope

This is a research and design record, not an accepted runtime change.  It defines how to
build an automatic SM120 quantization tuner for the existing Cube4 `piece_transformer`
while minimizing beam-ranking error.  No CUDA performance result is claimed here.  All
future kernel, throughput, ranking, frontier, and solve tests remain Molab-only.

## Exact subject

- Checkpoint: `test_results/cube4-model-bundle/model/model.pth`
- SHA-256: `58af301a4f2b77d503b6e12d450589c64c076624d3e1ff291128c23663ad3164`
- Checkpoint bytes: `13,555,825`
- Parameters: `3,383,064`
- Architecture: `seq_len=57`, `d_model=256`, `nhead=8`, four blocks,
  `ff_dim=1024`, ReLU, CLS pooling, `output_dim=24`, `num_classes=6`
- Matrix parameters studied: `3,368,192`

Parameter ownership:

| Class | Parameters | Initial precision policy |
|---|---:|---|
| Frontend embeddings + piece projection | 216,320 | FP16 initially |
| QKV projections | 786,432 | MXFP8 candidate |
| Attention output projections | 262,144 | MXFP8 candidate |
| FF1 projections | 1,048,576 | MXFP8 candidate |
| FF2 projections | 1,048,576 | MXFP8 candidate |
| Final output projection | 6,144 | FP16 |
| Norms and biases | 14,616 | FP16; FP32 reductions for normalization |

The 16 block GEMMs contain `3,145,728` weights, or 93.4% of all matrix weights.
This makes a mixed policy attractive: quantize the expensive block GEMMs and retain the
small, semantically sensitive frontend, normalization, residual, softmax, and final-logit
path in FP16/FP32.

## SM120 format contract

The native candidate is OCP MXFP8:

- E4M3 values;
- one E8M0 power-of-two scale per 32 consecutive K elements;
- dynamically quantized activation blocks;
- pre-quantized weight blocks;
- K extents divisible by 32;
- preferably FP32 accumulation with an FP16 output boundary, subject to the exact SM120
  CUTLASS instruction selected in the implementation.

All current Cube4 GEMM K extents (`256` and `1024`) satisfy block-32 alignment.  The
frontend piece projection also has K=768 but should remain FP16 in the first version because
it is a small share of work and converts discrete puzzle state into the model representation.

Primary format references:

- [NVIDIA Transformer Engine MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html)
- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
- [OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [CUTLASS SM120 MMA traits](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm120.hpp)

## Reproducible static weight findings

Method and full per-matrix results:

- `test_results/cube4_sm120_weight_quantization_analysis.py`
- `test_results/cube4_sm120_weight_quantization_stats_2026-08-22.json`

The script is CPU-only.  It uses PyTorch E4M3FN/E5M2 casts and simulates native MXFP8
block-32 E8M0 scaling.  It measures reconstruction error, not activation error or solve
quality.

| Weight representation | Aggregate NMSE | Aggregate SNR |
|---|---:|---:|
| E4M3, one power-of-two tensor scale | 0.000696921 | 31.568 dB |
| E4M3, per-row ideal amax scale | 0.000681873 | 31.663 dB |
| MXFP8 E4M3, block-32 E8M0 | 0.000696921 | 31.568 dB |
| Block-32 E4M3 with arbitrary ideal scales (non-native upper bound) | 0.000563503 | 32.491 dB |
| E5M2, one power-of-two tensor scale | 0.002754189 | 25.600 dB |

Important interpretation:

1. **E5M2 is rejected for forward weights.** It loses about 5.97 dB relative to E4M3.
2. **Weight-only E8M0 block scaling does not improve these weights by itself.** A
   power-of-two scale shifts the E4M3 exponent but does not add mantissa bits; the checkpoint
   has no static weight range problem requiring local rescue.
3. **The main MXFP8 accuracy question is activation distribution.** Dynamic block-32 scaling
   can protect activations from local outliers and underflow, while the SM120 instruction
   supplies the performance benefit.
4. Static MXFP8 SNR is very uniform across the expensive matrices: approximately
   31.56 dB for QKV, 31.57 dB for attention-out/FF1, and 31.59 dB for FF2.  Weight MSE alone
   cannot identify semantically sensitive layers.

### Correction experiments

- Keeping the top 1% individual E4M3 error entries in FP16 improves aggregate SNR only from
  31.57 to 32.53 dB.
- Keeping the top eight K columns of every matrix in FP16 uses 2.27% FP16 weights and reaches
  only 31.75 dB.
- Even the mathematically best rank-16 FP16 approximation of every core-MXFP8 residual uses
  8.33% correction parameters and reaches only 32.30 dB.

Therefore sparse or low-rank residual correction is **not a V1 default**.  It remains an
optional candidate only if end-to-end ranking tests reveal a concentrated failure that
static Frobenius error does not expose.

## Relevant industry and research approaches

| Approach | Useful idea | Adaptation for Cube4 | Decision |
|---|---|---|---|
| Native MXFP8 | E4M3 plus E8M0 scale per K-block of 32 | Direct hardware-compatible baseline | V1 foundation |
| SmoothQuant | Move activation-channel range into weights through an equivalent diagonal transform | Fold positive scales into LayerNorm affine parameters and adjacent weights | V1 search candidate |
| AWQ | Use activation statistics to identify salient weight channels | Rank sensitivity with real frontier activations; protect whole hardware-friendly matrices/blocks | Use principle, not its W4 kernel format |
| GPTQ | Use approximate second-order activation information to compensate weight rounding | Optimize E4M3 choices per K32 block using `X^T X`, if plain MXFP8 misses quality | V2 candidate |
| QuaRot / SpinQuant | Rotate hidden dimensions to remove outliers | Only use transformations that can be folded into weights or fused cheaply; preserve QK dot products | V2 candidate |
| SpQR | Isolate outlier weights in high precision | Static Cube4 result shows weak return and irregular sparse kernels risk losing speed | Not default |
| Mixed precision sensitivity | Roll sensitive layers back to FP16 | Natural fit: only 16 expensive matrices and a small model | V1 mandatory |
| QAT | Retrain with fake quantization | Last resort when PTQ cannot meet ranking/solve gates | V3 fallback |

Primary papers:

- [SmoothQuant, ICML 2023](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf)
- [AWQ, MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [QuaRot, NeurIPS 2024](https://openreview.net/attachment?id=8CRc1w7Y28&name=pdf)
- [SpQR, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1787533e171dcc8549cc2eb5a4840eec-Abstract-Conference.html)
- [PTQ with microscaling formats](https://arxiv.org/abs/2405.07135)
- [2026 MXFP PTQ benchmark](https://arxiv.org/abs/2601.09555)

The 2026 MXFP study reports MXFP8 as consistently near-lossless on its tested LLMs, but that
is evidence for prioritizing MXFP8, not evidence that this Cube4 ranking model is already
safe.  Beam threshold decisions require their own validation.

## Cube4-specific graph-preserving transformations

The tuner must never apply generic LLM transforms blindly.  It may search only transformations
that preserve the FP16 graph before quantization:

1. **LayerNorm → QKV:** positive per-channel scales can be folded into LayerNorm gamma/beta
   and the inverse scale into QKV weight columns.
2. **LayerNorm → FF1:** the same fold is valid for the FF branch.
3. **FF1 → ReLU → FF2:** positive diagonal scaling can be moved through ReLU because ReLU is
   positively homogeneous; compensate in FF2 columns.
4. **Q/K:** reciprocal per-head-dimension scaling (`Q*S`, `K/S`) preserves the dot product
   before quantization.  Shared orthogonal rotations preserve QK dot products but must be
   folded without adding an unfused runtime kernel.
5. **V → attention mix → output projection:** V-channel scaling can be inversely folded into
   attention-output weight columns because the attention-weighted sum is linear in V.
6. **Residual stream:** do not rotate or rescale the residual representation unless the
   transformation and its inverse are completely folded across every consumer.

## Proposed automatic tuner

### 1. Deterministic inputs and fingerprints

Record checkpoint SHA, model metadata SHA, generator SHA, solver commit, CUDA/CUTLASS versions,
GPU PCI identity, seed set, and calibration-corpus hash.  A profile is invalid if any of these
change.

### 2. Calibration corpus

Use separate calibration and holdout sets, stratified as follows:

- solved target and shallow random walks;
- training-like random walks at depths 2–45;
- deeper walks at 46–100;
- **real FP16 beam-frontier reservoir samples from every depth 4–8**;
- extra samples around the beam cutoff and states with small top-move margins;
- deterministic duplicate removal and per-depth quotas.

Random inputs alone are prohibited for quality decisions.  Real frontiers are required because
the model changes its own future input distribution through beam selection.

### 3. Statistics to collect

For every quantizable GEMM and calibration stratum:

- activation and weight amax/RMS/percentiles;
- E8M0 block-scale exponent histogram;
- saturation, underflow-to-zero, and subnormal rates;
- FP16 versus MXFP8 output error;
- QK score and softmax-distribution divergence;
- final 24-logit absolute and relative errors;
- top-1 agreement, top-k overlap, and pair inversions;
- errors specifically for candidates near the global beam threshold.

### 4. Candidate family

Start with a deliberately small, hardware-realizable search space:

- each of the 16 core matrices: MXFP8 or FP16;
- activation: dynamic MXFP8 block-32 or FP16;
- graph-aware SmoothQuant alpha grid per branch;
- optional reciprocal Q/K balance grid;
- output projection, norms, residuals, softmax, biases: fixed high precision;
- no sparse or low-rank correction in the initial sweep.

### 5. Search strategy

1. All-FP16 baseline.
2. All-core-MXFP8 baseline.
3. Quantize one matrix at a time to measure marginal sensitivity.
4. Starting from all-core-MXFP8, greedily roll the most damaging matrix back to FP16.
5. Coordinate-search graph-preserving scale parameters on the calibration set.
6. Re-score the Pareto frontier on the holdout set.
7. Benchmark only numerically accepted candidates on SM120.

The model has only 16 expensive matrices, so the tuner should retain a Pareto set instead of
hiding accuracy and speed in one arbitrary scalar score.

### 6. Quality objectives

Tensor MSE is diagnostic only.  The optimization hierarchy is:

1. finite outputs and deterministic replay;
2. minimize top-24 ranking inversions;
3. maximize exact top-1 and top-k agreement;
4. maximize frontier Jaccard overlap per depth;
5. minimize changes among candidates within a narrow band of the beam threshold;
6. retain solve success and acceptable solution length;
7. among accepted candidates, minimize exact `depth_done=8` wall time.

Initial acceptance thresholds should be recorded as tunable policy rather than treated as
facts.  A sensible first strict profile is top-1 agreement >=99.9%, top-24 set overlap >=99.9%,
and per-depth frontier Jaccard >=99.5% on holdout, followed by a real solve gate.  These numbers
must be revised from empirical correlation with solve behavior.

### 7. Molab acceptance funnel

1. Unit test the exact quantize/dequantize bit behavior against a trusted reference.
2. Compare layer dumps and full 24-score dumps on holdout states.
3. Benchmark isolated real Cube4 Stream1 at the accepted microbatch/concurrency.
4. Run beam `2**16` through exact `depth_done=8`, compare full frontier hashes/statistics.
5. Run beam `2**25` through exact `depth_done=8`; report time, utilization, VRAM, and Stream3 jobs.
6. Run the agreed full puzzle solve before promoting a public profile.

Every CUDA build and all GPU evidence in this funnel must run on Molab, not locally.

## Output contract

The tuner should emit an immutable directory rather than mutate the checkpoint:

```text
sm120_quant_profile/
  manifest.json
  weights_mxfp8/
  scales_e8m0/
  folded_fp16/
  calibration_manifest.json
  layer_metrics.jsonl
  ranking_metrics.json
  frontier_metrics.json
  benchmark_metrics.json
  profile.json
```

`profile.json` must explicitly describe each operator's weight dtype, activation dtype,
accumulator/output dtype, scale granularity, scale orientation, folded transforms, and fallback
precision.  It must include hashes of every artifact and never silently fall back to FP16.

## Recommended V1

Implement and test this order when development is authorized:

1. native MXFP8 E4M3/E8M0 for the 16 core block GEMMs;
2. dynamic activation quantization per K32 block;
3. FP32 accumulation where supported, FP16 output boundary;
4. keep frontend, normalization, residual, softmax, and final output in the established path;
5. add real-frontier activation capture and ranking-aware evaluation before scale tuning;
6. add graph-folded SmoothQuant/equalization;
7. use mixed FP16 rollback for sensitive matrices;
8. consider Hessian-aware floating-point rounding only if the preceding steps miss quality.

The static checkpoint evidence rejects E5M2 and does not justify sparse/low-rank correction.
The largest remaining uncertainty is activation quantization on actual beam-selected states.
