#!/usr/bin/env bash
set -euo pipefail
export BEAM_STREAM1_TRANSFORMER_HOPPER_QKV=fp16_tma
export BEAM_STREAM1_TRANSFORMER_HOPPER_FF1=fp16_tma
export BEAM_STREAM1_TRANSFORMER_HOPPER_FF1_EPILOGUE=128x64
export BEAM_STREAM1_TRANSFORMER_FF2_POLICY=m128n128
export BEAM_STREAM1_TRANSFORMER_COMPACT57=1
export BEAM_B_MICRO=384 BEAM_STREAM1_CONCURRENCY=8
export BEAM_HOPPER_NATIVE_FP8=1
export LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64
export PUZZLE_ID=1000 DEPTH_LIMIT=9 BEAM_WIDTH=100000000
mkdir -p /workspace/fp8_results
for mode in fp16 fp8; do
  if [[ $mode == fp16 ]]; then export BEAM_WEIGHT_DIR=/workspace/stream1_transformer_weights_fp16
  else export BEAM_WEIGHT_DIR=/workspace/hopper_offline_qkv_e4m3_v2; fi
  export BEAM_HISTORY_DIR=/workspace/fp8_pipeline_history_${mode}
  # Timing runs may return unsolved. Preserve exit code without hiding errors.
  set +e
  bash /workspace/MGBFS/hpc/h200_single_pipeline.sh > /workspace/fp8_results/pipeline_${mode}.log 2>&1
  rc=$?
  set -e
  printf 'exit_code=%s\n' "$rc" >> /workspace/fp8_results/pipeline_${mode}.log
  if [[ $rc != 0 && $rc != 2 ]]; then exit "$rc"; fi
done
