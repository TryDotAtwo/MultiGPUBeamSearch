# Molab Cube4 compact57 and final CLS-attention validation

Date: 2026-08-22

## Contract

- All CUDA compilation, correctness checks, and benchmarks ran on Molab.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, SM120.
- Model: real Cube4 fp16 piece Transformer, sequence length 57, d_model 256,
  four layers, `output_dim=24`.
- Isolated workload: 896 parents per Transformer microbatch, concurrency 4,
  CUDA Graph replay, independent processes.
- Integrated workload: competition `cayley-py-444-cube`, puzzle 0, beam
  `2**25`, runner limit 9; acceptance timing is exactly `depth_done=8`.

## Compact sequence

Padded sequence 64 and compact sequence 57 produced byte-identical full score
dumps (SHA-256 `8a613833f7e029ad577ec42031accaa21bc926bc366380413751e17f5bab2859`).
Compact scratch fell from 1,409,458,176 to 1,232,437,248 bytes (12.56%).

Nine-process isolated graph replay:

| Shape | Median latency | Parents/s | Decision |
|---|---:|---:|---|
| padded64 + final CLS | 9.8455 ms | 364,023 | control |
| compact57 + final CLS | 8.6338 ms | 415,113 | accept (1.1403x) |

## GEMM and attention policy sweep

All candidates retained the same score-key digest
`13477114594214371836`. Larger QKV/FF1/attention-output/FF2 policies were
slower. Attention `exact32` appeared slightly faster in a two-run screen, but a
nine-process gate measured 8.6475 ms versus auto 8.6404 ms, so it was rejected.

## Generic final CLS-only attention

The existing opt-in Q=1 CUTLASS FMHA path was connected to the generic final
CLS implementation. The 896-row exactness test passed. Nine independent graph
replay processes produced the same digest for every profile:

| Final-layer attention | Median | Mean | Range | Speedup |
|---|---:|---:|---:|---:|
| full queries | 8.6331 ms | 8.6226 ms | 8.5868-8.6809 | 1.0000x |
| CLS Q=1 default tile | 8.6137 ms | 8.6357 ms | 8.5661-8.7296 | 1.0023x |
| CLS Q=1 `q32k64` | **8.4919 ms** | 8.4997 ms | 8.4550-8.5686 | **1.0166x** |

Relative to the original padded/full-final control median 12.2824 ms, the
accepted compact57 + final-CLS + CLS-Q=1 path is about 1.4464x faster in the
isolated Stream1 graph benchmark.

## Integrated beam 2^25 depth 8

The terminal detached Molab run used external parent batch 3584, Transformer
microbatch 896, concurrency 4, 24 Stream3 accumulation slots, two graph-window
rings (12 graph executables per lane, 48 physical templates), and final
materialization chunk 88,064. Host history used disk mode so GPU measurements
were not coupled to a growing host-RAM cache.

| Depth | Input frontier | Time | Stream3 jobs | Next frontier |
|---:|---:|---:|---:|---:|
| 6 | 18,426,554 | 46.7675 s | 215 | 33,554,432 |
| 7 | 33,554,432 | 84.1182 s | 391 | 33,554,432 |
| **8** | **33,554,432** | **84.2199 s** | **391** | **33,554,432** |

The runner completed normally in 221.081 seconds with no OOM, overflow, fatal,
or illegal-access evidence. Observed steady VRAM was about 19.4 GiB. The prior
accepted SM120 auto-default measured 111.024 seconds at exactly depth 8 on the
same beam/model/GPU workload, so the accepted integrated profile is 1.3183x
faster at the required acceptance point.

The first attempt intentionally demonstrated the fail-closed graph window
guard: `2` was incorrectly interpreted as templates per lane and rejected
because one 24-slot ring needs `ceil(24/4)=6` templates per lane. The accepted
value `12` therefore represents exactly two inference windows.

## Warm stage profile

An opt-in, eager-only CUDA-event profiler measured the second launch after one
warm-up call, avoiding CUTLASS lazy-initialization noise. The production graph
path is unchanged; enabling the profiler during CUDA Graph capture fails closed
with an explicit error.

| Stage group | Three full layers | Share of 1.97635 ms |
|---|---:|---:|
| LN1 + LN2 | 0.272384 ms | 13.78% |
| QKV GEMM | 0.238848 ms | 12.09% |
| attention | 0.351008 ms | 17.76% |
| attention output | 0.133952 ms | 6.78% |
| FF1 + FF2 | **0.739168 ms** | **37.40%** |
| final CLS layer | 0.240992 ms | 12.19% |

The measured total was 1.97635 ms. The surrounding eager benchmark reported
2.2435 ms; the normal no-profiler CUDA Graph control reported 1.9911 ms,
10,800,096.7 candidates/s, checksum `402305984`, and score-key digest
`2176418464504111356`. Therefore the next optimization target is the FFN and
its epilogue/dataflow rather than another attention tile sweep.

The negative control returned non-zero as designed with
`BEAM_STREAM1_TRANSFORMER_STAGE_PROFILE requires eager execution; CUDA Graph
capture is active`. The no-profiler graph control returned zero and retained
the expected checksum and digest.
