# Cube4 H200 single-GPU measured profile

Measured on Vast H200 SXM, CUDA 12.8, CUTLASS v3.9.2. This is a session-specific
reproduction profile, not a hardware-independent default or an FP8 backend.

Build with `BEAM_CUDA_ARCHITECTURES=90a`, `BEAM_STATE_LOGICAL_BYTES=96`, and the
Cube4 puzzle-info file. See `test_results/h200_single_2026-09-15.md` for evidence
and remaining quality limits. All model weights must retain their original
manifest-defined ReLU activation and output-24 move ordering.

The scripts expect repository `/workspace/MGBFS`, binaries `/workspace/build`,
Cube4 exported weights and manifest directly under `/workspace`, and
`/workspace/test.csv` plus `/workspace/puzzle_info.json`. Copy the relevant
scripts together to `/workspace` before running. They write to
`/workspace/results` and do not publish solutions.

Measured optimized profile:

```bash
export BEAM_STREAM1_TRANSFORMER_HOPPER_QKV=fp16_tma
export BEAM_STREAM1_TRANSFORMER_HOPPER_FF1=fp16_tma
export BEAM_STREAM1_TRANSFORMER_FF2_POLICY=m128n128
export BEAM_STREAM1_TRANSFORMER_COMPACT57=1
export BEAM_B_MICRO=384
export BEAM_STREAM1_CONCURRENCY=8
bash /workspace/h200_single_optimized.sh
```

The wrapper defaults to puzzle 1000, depth limit 9, beam 100M; override
`PUZZLE_ID`, `DEPTH_LIMIT`, `BEAM_WIDTH` to change them. A timing run ending
unsolved is not a solution-quality certificate.

Keep `COMPACT57` unchanged from scratch allocation through graph replay.
It changes both token storage stride and scratch sizing. Attention auxiliary
scratch remains conservatively sized for the old padded sequence. Omit/unset
the flag to retain the old 64-token layout. No persistent frontier state
padding contract changes.

Validation scripts:

- `h200_combined_sweep.sh`: padded layout, combined GEMM policies.
- `h200_compact_sweep.sh`: compact layout, paired score dumps and timings.
- `h200_validate_compact.sh`: original versus optimized CSV-seeded scores and
  a small memcheck; six first-sticker variants per puzzle are not a broad
  reachable-frontier corpus.
- `h200_single_profile.sh`: bounded Nsight trace; set `H200_PROFILE_TAG` to a
  distinct name for each capture. Headline rates use unprofiled runs.

`h200_budget_stop.sh` belongs specifically to Vast instance 51111383 and must
not be reused for another rental. Stop retains paid disk storage; destroying
the instance requires first archiving needed artifacts.
