#!/usr/bin/env bash
set -euo pipefail
export H200_WORLD_SIZE=8
export BEAM_WIDTH=256000000
export H200_PROFILE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles/h200_8gpu_cube4_256m.json"
# Historical 2026-09-23 template, not acceptance of the current binary.
# Outer parent batch and inner model microbatch are independent knobs.
export BEAM_B_MICRO=${BEAM_B_MICRO:-2048}
export BEAM_STREAM1_TRANSFORMER_MICRO=${BEAM_STREAM1_TRANSFORMER_MICRO:-1024}
export BEAM_STREAM1_CONCURRENCY=${BEAM_STREAM1_CONCURRENCY:-4}
export BEAM_STREAM3_RING_SLOTS=${BEAM_STREAM3_RING_SLOTS:-8}
export BEAM_SHARD_COUNT=${BEAM_SHARD_COUNT:-64}
export BEAM_STREAM4_BATCH_CANDIDATES=${BEAM_STREAM4_BATCH_CANDIDATES:-262144}
export BEAM_STREAM4_TRIGGER_CANDIDATES=${BEAM_STREAM4_TRIGGER_CANDIDATES:-393216}
echo 'profile_template=h200_8gpu_256m_20260923 current_binary_quality=unaccepted'
exec bash "$(dirname "${BASH_SOURCE[0]}")/h200_dual_large_beam.sh"
