#!/usr/bin/env bash
# Validated 1xH200 Cube4 profile; requires source d10c6775 or newer.
# Build directory defaults to the physically tested build, override explicitly.
set -euo pipefail
export H200_BUILD_DIR=${H200_BUILD_DIR:-/workspace/build_ffn}
export BEAM_WEIGHT_DIR=${BEAM_WEIGHT_DIR:-/workspace/stream1_transformer_weights_fp16}
export BEAM_STREAM1_TRANSFORMER_HOPPER_QKV=fp16_tma
export BEAM_STREAM1_TRANSFORMER_HOPPER_FF1=fp16_tma
export BEAM_STREAM1_TRANSFORMER_HOPPER_FF2=fp16_tma
export BEAM_STREAM1_TRANSFORMER_HOPPER_FF1_EPILOGUE=128x64
export BEAM_STREAM1_TRANSFORMER_FF2_POLICY=m128n128
export BEAM_STREAM1_TRANSFORMER_COMPACT57=1
export BEAM_STREAM1_TRANSFORMER_DUAL_INPUT_LN=0
export BEAM_B_MICRO=384 BEAM_STREAM1_CONCURRENCY=8
unset BEAM_HOPPER_NATIVE_FP8
export LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64
exec bash "$(dirname "$0")/h200_single_pipeline.sh"
