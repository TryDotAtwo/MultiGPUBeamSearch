#!/usr/bin/env bash
set -euo pipefail
cd /workspace/MGBFS
unset BEAM_STREAM1_SYNTHETIC_STATES
export BEAM_WEIGHT_DIR=/workspace
export BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1 BEAM_STREAM1_TRANSFORMER_BENCH_ITERS=10
export BEAM_STREAM1_TRANSFORMER_B_MICRO=384 BEAM_STREAM1_TRANSFORMER_CONCURRENCY=8
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1 BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1 BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1
for puzzle in 1 2 3 4 5 6 7 8 9 10 1000; do
  for variant in baseline optimized; do
    if [[ $variant == baseline ]]; then
      export BEAM_STREAM1_TRANSFORMER_HOPPER_QKV=off BEAM_STREAM1_TRANSFORMER_HOPPER_FF1=off
      unset BEAM_STREAM1_TRANSFORMER_FF2_POLICY
      export BEAM_STREAM1_TRANSFORMER_COMPACT57=0
    else
      export BEAM_STREAM1_TRANSFORMER_HOPPER_QKV=fp16_tma BEAM_STREAM1_TRANSFORMER_HOPPER_FF1=fp16_tma
      export BEAM_STREAM1_TRANSFORMER_FF2_POLICY=m128n128 BEAM_STREAM1_TRANSFORMER_COMPACT57=1
    fi
    BEAM_STREAM_BENCH_REPORT=/workspace/results/quality_p${puzzle}_${variant}.md \
      BEAM_STREAM1_TRANSFORMER_SCORE_DUMP=/workspace/results/quality_p${puzzle}_${variant}.bin \
      /workspace/build/stream_benchmark "$puzzle" > /workspace/results/quality_p${puzzle}_${variant}.log 2>&1
  done
  cmp /workspace/results/quality_p${puzzle}_baseline.bin /workspace/results/quality_p${puzzle}_optimized.bin
  echo "quality_p${puzzle}=exact"
done
export BEAM_STREAM1_TRANSFORMER_B_MICRO=128 BEAM_STREAM1_TRANSFORMER_CONCURRENCY=1
export BEAM_STREAM1_TRANSFORMER_BENCH_ITERS=1 BEAM_STREAM_BENCH_REPORT=/workspace/results/compact_memcheck.md
compute-sanitizer --tool memcheck --error-exitcode=99 /workspace/build/stream_benchmark 1000 > /workspace/results/compact_memcheck.log 2>&1
