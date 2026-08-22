# Molab Cube4 final-CLS split-QKV validation

Date: 2026-08-22

## Contract

- Every CUDA build, correctness check, microbenchmark, and integrated solve ran
  on Molab, not on the local GPU.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120.
- Model: real Cube4 fp16 Piece Transformer, sequence length 57, d_model 256,
  four layers, FF width 1024, output_dim 24.
- Integrated acceptance point: puzzle 0, beam `2**25`, exactly
  `depth_done=8`.

## Implemented path

The final CLS-only layer now has an opt-in split-QKV route controlled by
`BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1`. It computes Q only for the
896 CLS rows and K/V for all 896 x 57 token rows, while preserving the
interleaved exported QKV layout. The path requires the existing generic
final-CLS attention mode, fp16, d_model 256, and SM80 or newer. The default
path remains unchanged when the flag is absent.

The 896-parent CUDA exactness test passed byte-for-byte. All isolated runs
retained score-key digest `13477114594214371836`.

## Isolated A/B

Eleven independent CUDA Graph processes used 896 parents, concurrency 4,
compact sequence 57, final CLS attention, and `q32k64`.

| Path | Median | Mean | Decision |
|---|---:|---:|---|
| Control full QKV | 8.4660 ms | 8.47651 ms | control |
| Split final Q + full K/V | **8.3784 ms** | **8.39359 ms** | accept |

Median speedup was 1.01046x. Warm eager stage profiling showed the final CLS
layer falling from 0.240992 ms to 0.226624 ms and total measured Transformer
time from 1.97635 ms to 1.95379 ms.

## Integrated beam 2^25 depth 8

The accepted profile used external parent batch 3584, Transformer microbatch
896, concurrency 4, 24 Stream3 accumulation slots, 12 graph templates per
lane (two windows), 16 shards, and final materialization chunk 88,064. History
was stored in disk mode. The run returned zero and did not report OOM,
overflow, fatal, or illegal-access failures.

| Depth | Frontier | Time | Stream3 jobs |
|---:|---:|---:|---:|
| 6 | 33,554,432 | 80.0725 s | 391 |
| 7 | 33,554,432 | 80.0780 s | 391 |
| **8** | **33,554,432** | **80.2952 s** | **391** |

The previous accepted control was 84.2199 seconds at depth 8. The integrated
speedup is therefore 1.04888x (4.66% lower wall time). Total process wall time
was 255.026 seconds for this data snapshot.

## Rejected alternatives

- Replacing the FFN GEMMs with raw cuBLAS was rejected after Molab screens:
  FF1 did not include the currently fused bias+SiLU work and FF2 was slower
  than the CUTLASS path.
- A fused FF2 residual epilogue was reverted because exactness failed. The
  established path computes LayerNorm statistics from residual+bias before
  fp16 materialization; the attempted CUTLASS epilogue rounded to fp16 first
  and therefore changed arithmetic semantics.
