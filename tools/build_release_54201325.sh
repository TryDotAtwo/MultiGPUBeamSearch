#!/usr/bin/env bash
set -eu
cd /workspace
git clone --quiet https://github.com/NVIDIA/cutlass.git cutlass
git -C cutlass checkout --quiet 0b55a2f691d69981583568fd9eb69687b1f0de8a
git -C cutlass rev-parse HEAD
cd /workspace/release_audit_54201325
/venv/main/bin/python -m pip install pytest-subtests pandas jsonschema > /workspace/release_dependencies_54201325.log 2>&1
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DBEAM_CUDA_ARCHITECTURES=86 \
  -DBEAM_STATE_LOGICAL_BYTES=96 -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=24 \
  -DCUTLASS_DIR=/workspace/cutlass -DBEAM_ENABLE_DEBUG=ON -DBEAM_ENABLE_DEPTH_LOGS=ON \
  -DBEAM_DEBUG_STREAM_TIMING=OFF > /workspace/release_configure_54201325.log 2>&1
cmake --build build-release --target production_runner -j 8 > /workspace/release_build_54201325.log 2>&1
build-release/production_runner --build-info
sha256sum build-release/production_runner
CUTLASS_DIR=/workspace/cutlass PYTHONPATH=. /venv/main/bin/python -m pytest --ignore=tests/cayleypy_public -q > /workspace/release_regression_54201325.log 2>&1
tail -n 8 /workspace/release_regression_54201325.log
