# Cube4 SM120 CUDA GEMM auto-policy benchmark

Date: 2026-08-21  
Starting commit: `d319f6db0540b6ccef0d41258407815cfe163cce`

## Contract

- NVIDIA RTX PRO 6000 Blackwell Server Edition (`SM120`)
- Cube4 puzzle 0, fp16 piece Transformer, `output_dim=24`
- Beam `2**25`, runner limit 9, acceptance metric exactly `depth_done=8`
- External `B_MICRO=3584`, Transformer microbatch 896, concurrency 4
- Two inference rings, 24 Stream3 accumulation slots, 391 Stream3 jobs/full layer

## Isolated Stream1 CUDA-event A/B

Five independent process runs per profile, each using CUDA Graph replay and the
same score checksum/digest.

| Profile | Median group latency | Parents/s | Digest |
|---|---:|---:|---:|
| Baseline | 13.3594 ms | ~268,300 | 8641242062488159533 |
| SM120 candidate | 11.8302 ms | ~302,950 | 8641242062488159533 |

The candidate uses `m128n128` for QKV, FF1, FF2, and attention-out, plus FF1
identity-4 swizzle. Median latency improved 11.45% and throughput about 12.93%.
Persistent LayerNorm was rejected because it changed the score digest. Attention
tile, exact32, extra swizzles, two-stage FF1, and fused residual epilogues did not
improve the selected candidate.

## Full search A/B

| Profile | Depth 6 | Depth 7 | Depth 8 | Stream3 jobs | Result |
|---|---:|---:|---:|---:|---|
| Baseline CUDA | 124.118 s | 124.256 s | 124.367 s | 391 | rc=0 |
| Explicit candidate env | 110.978 s | 111.066 s | 111.109 s | 391 | rc=0 |
| New SM120 auto-default, no policy env | 110.933 s | 110.917 s | **111.024 s** | 391 | rc=0 |

The production auto-default reduced exact depth-8 time by 10.73%. No OOM or
overflow was observed. Explicit environment values still override the automatic
selection, and architectures below SM120 retain the previous baseline defaults.

## Verification

- TDD RED: policy test failed to compile because the selector functions did not exist.
- TDD GREEN: `stream1_transformer_gemm_policy_tests=pass`.
- Build: `stream_benchmark` and `production_runner` linked successfully.
- Relevant test suite: 4/4 passed (`gemm_policy`, `attention_policy`, `shape`,
  `layernorm_policy`).
- New no-env benchmark checksum equals explicit baseline checksum.

