# SM120 Cube4 mixed-precision Molab progress — 2026-08-22

## Contract

- Quality truth: original FP32 checkpoint.
- Current production reference: immutable FP16 export.
- Candidate: immutable offline-encoded mixed profile. Production performs no weight quantization.
- GPU compile and runtime verification: Molab RTX PRO 6000 Blackwell Server Edition, SM120, CUDA 13.
- Search workload: Cube4 output_dim=24. Target benchmark is beam `2**25`, completed depth 8.

## Implemented and pushed

- `67f7eeb1`: native Stream1 now propagates the manifest activation and executes Cube4 ReLU rather than hard-coded SiLU.
- `a4e252d3`, `7f2724b1`: deterministic offline per-128x128 MSE scale search and immutable E4M3 artifact encoding.
- `021b251a`: frontier gate reconstructs actual puzzle states from history. It no longer compares CandidateMeta bytes; it supports both public `moves`/`actions` generator schemas and comma/semicolon state CSVs.

## Molab verification

- Native Cube4 runner compiled for state_len=96, move_count=24, sm_120a.
- `contract_tests=pass`.
- Python targeted suite: `9 passed`.
- CUDA primitive test: `nmse=0.000415495`, `max_abs_error=0.0268555`.
- Immutable MSE artifact created at
  `/marimo/storage/cayleypy-cube4/sm120_profiles/block0_ff1_mse_1787394780`.

## Quality observations

Correct ReLU FP16 versus one-pass block0 FF1 E4M3 at beam `2**16`:

- FP16 solver time: `1.83072 s`.
- max-abs E4M3: `1.84262 s`.
- MSE-scaled E4M3: `1.84414 s`.
- Reconstructed-state Jaccard for MSE profile: depth 3 `0.29079`, depth 4 `0.03978`, depth 8 `0.0002366`.
- Therefore neither naïve nor weight-MSE E4M3 is eligible for the target benchmark.

Weight-stripe rollback showed activation quantization is the dominant loss: restoring all FF1 N-stripes to FP16 while retaining E4M3 activations did not restore ranking quality.

Two-component E4M3 diagnostics (2048 real holdout states) improved mixed-vs-FP16 quality:

- 3-term `A_hi B_hi + A_hi B_lo + A_lo B_hi`: top1 `0.99756`, top8 `0.99750`, global overlap `0.99707`, RMSE `0.00326` in the first diagnostic.
- FP32-output-accumulation variant: top1 `0.99658`, top8 `0.99780`, global overlap `0.99658`, RMSE `0.00327`.
- These remain below the configured `0.999` ranking gates and were not promoted.

## Correct target corpus

Clarification recorded 2026-08-24: the user confirmed that the completed
80.2952 s run is a correct ReLU Cube4 result. The earlier attribution of that
run to SiLU was inferred from commit chronology and is withdrawn; chronology
did not prove the effective Molab checkout or uncommitted runtime patch. A
separate detached ReLU FP16 `2**25` run was started under
`/marimo/storage/cayleypy-cube4/relu_fp16_target_2p25/1787395409` and persisted
histories through at least depth 5, but the sandbox expired with HTTP 410. That
partial run does not replace or invalidate the accepted completed baseline.

## Remaining gate

1. Start a fresh Molab sandbox and regenerate the complete ReLU FP16 target corpus through depth 8.
2. Complete the block-scaled INT8 diagnostic (higher precision than E4M3 with INT32 accumulation) on that corpus.
3. Implement only a candidate that passes FP32/FP16 ranking gates and reconstructed-state frontier gate at `2**16`.
4. Benchmark accepted candidates at `2**25` against the accepted correct-ReLU
   FP16 baseline of 80.2952 s, with exactly `depth_done=8` and 391 Stream3 jobs.
