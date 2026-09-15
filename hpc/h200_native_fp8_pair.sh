#!/usr/bin/env bash
set -euo pipefail
cd /workspace/hopper_runtime
result_dir=${H200_RESULTS_DIR:-/workspace/fp8_results}
build_dir=${H200_BUILD_DIR:-/workspace/build}
fp8_artifact=${H200_FP8_ARTIFACT:-/workspace/hopper_offline_qkv_e4m3_v2}
mkdir -p "$result_dir"
export BEAM_STREAM1_SYNTHETIC_STATES=1 BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1
export BEAM_STREAM1_TRANSFORMER_BENCH_ITERS=300
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1 BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1 BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1
export BEAM_STREAM1_TRANSFORMER_HOPPER_QKV=fp16_tma BEAM_STREAM1_TRANSFORMER_HOPPER_FF1=fp16_tma
export BEAM_STREAM1_TRANSFORMER_FF2_POLICY=m128n128 BEAM_STREAM1_TRANSFORMER_COMPACT57=1
export BEAM_STREAM1_TRANSFORMER_B_MICRO=384 BEAM_STREAM1_TRANSFORMER_CONCURRENCY=8
export BEAM_STREAM1_TRANSFORMER_DUAL_INPUT_LN=0 BEAM_STREAM1_TRANSFORMER_HOPPER_FF1_EPILOGUE=128x64
export BEAM_HOPPER_NATIVE_FP8=1
export LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64
run_one(){
  local mode=$1 tag=$2 puzzle=$3
  if [[ $mode == fp16 ]]; then export BEAM_WEIGHT_DIR=/workspace/stream1_transformer_weights_fp16
  else export BEAM_WEIGHT_DIR=$fp8_artifact; fi
  BEAM_STREAM_BENCH_REPORT=${result_dir}/${tag}.md \
  BEAM_STREAM1_TRANSFORMER_SCORE_DUMP=${result_dir}/${tag}.bin \
  "${build_dir}/stream_benchmark" "$puzzle" > "${result_dir}/${tag}.log" 2>&1
}
for rep in 1 2 3 4 5 6 7; do
  order='fp16 fp8'; if (( rep % 2 == 0 )); then order='fp8 fp16'; fi
  for mode in $order; do run_one "$mode" "native_${mode}_${rep}" 1000; done
done
unset BEAM_STREAM1_SYNTHETIC_STATES
export BEAM_STREAM1_TRANSFORMER_BENCH_ITERS=3
for puzzle in 1 2 3 4 5 6 7 8 9 10 1000; do
  for mode in fp16 fp8; do run_one "$mode" "quality_p${puzzle}_${mode}" "$puzzle"; done
done
