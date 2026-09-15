#!/usr/bin/env bash
set -euo pipefail
cd /workspace/MGBFS
export BEAM_WEIGHT_DIR=/workspace BEAM_STREAM1_SYNTHETIC_STATES=1
export BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1 BEAM_STREAM1_TRANSFORMER_BENCH_ITERS=300
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1 BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1 BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1
export BEAM_STREAM1_TRANSFORMER_HOPPER_QKV=fp16_tma BEAM_STREAM1_TRANSFORMER_HOPPER_FF1=fp16_tma
export BEAM_STREAM1_TRANSFORMER_FF2_POLICY=m128n128 BEAM_STREAM1_TRANSFORMER_COMPACT57=1
export BEAM_STREAM1_TRANSFORMER_B_MICRO=384 BEAM_STREAM1_TRANSFORMER_CONCURRENCY=8
for rep in 1 2 3 4 5; do
  for mode in 0 1; do
    tag=dual_ln_mode${mode}_${rep}
    BEAM_STREAM1_TRANSFORMER_DUAL_INPUT_LN=$mode \
      BEAM_STREAM_BENCH_REPORT=/workspace/results/${tag}.md \
      BEAM_STREAM1_TRANSFORMER_SCORE_DUMP=/workspace/results/${tag}.bin \
      /workspace/build/stream_benchmark 1000 > /workspace/results/${tag}.log 2>&1
  done
  cmp /workspace/results/dual_ln_mode0_${rep}.bin /workspace/results/dual_ln_mode1_${rep}.bin
done
unset BEAM_STREAM1_SYNTHETIC_STATES
export BEAM_STREAM1_TRANSFORMER_BENCH_ITERS=3
for puzzle in 1 2 3 4 5 6 7 8 9 10 1000; do
  for mode in 0 1; do
    tag=dual_ln_quality_p${puzzle}_mode${mode}
    BEAM_STREAM1_TRANSFORMER_DUAL_INPUT_LN=$mode \
      BEAM_STREAM_BENCH_REPORT=/workspace/results/${tag}.md \
      BEAM_STREAM1_TRANSFORMER_SCORE_DUMP=/workspace/results/${tag}.bin \
      /workspace/build/stream_benchmark "$puzzle" > /workspace/results/${tag}.log 2>&1
  done
  cmp /workspace/results/dual_ln_quality_p${puzzle}_mode0.bin /workspace/results/dual_ln_quality_p${puzzle}_mode1.bin
  echo "dual_ln_quality_p${puzzle}=exact"
done
