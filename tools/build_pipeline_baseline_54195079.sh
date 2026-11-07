#!/usr/bin/env bash
set -eu
cd /workspace
git clone --quiet https://github.com/NVIDIA/cutlass.git cutlass
git -C cutlass checkout --quiet 0b55a2f691d69981583568fd9eb69687b1f0de8a
git -C cutlass rev-parse HEAD
tar -xzf vast_paired_fixture_20261002.tar.gz -C /workspace
/venv/main/bin/python -m pip install --no-deps --target /workspace/nccl225_54195079 \
  nvidia-nccl-cu12==2.25.1 > /workspace/nccl225_install_54195079.log 2>&1
cd /workspace/pipeline_audit_54195079
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBEAM_CUDA_ARCHITECTURES=86 \
  -DBEAM_STATE_LOGICAL_BYTES=96 -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=24 \
  -DCUTLASS_DIR=/workspace/cutlass -DBEAM_ENABLE_DEBUG=ON -DBEAM_ENABLE_DEPTH_LOGS=ON \
  -DBEAM_DEBUG_STREAM_TIMING=ON > /workspace/pipeline_configure_54195079.log 2>&1
cmake --build build --target production_runner stream4_cuda_tests final_union_dispatcher_cuda_tests -j 8 \
  > /workspace/pipeline_baseline_build_54195079.log 2>&1
mkdir -p build-before evidence/recovered_campaign_inputs_20261001
cp build/production_runner build-before/production_runner
sha256sum build-before/production_runner
ln -s /workspace/cube4_paired_reference_20261001 evidence/cube4_paired_reference_20261001
cp /workspace/dual_base_profile_54186691.json evidence/dual_base_profile.json
cp /workspace/run_vast_dual_neural_20261002.py /workspace/run_vast_reflected_20261002.py \
   /workspace/check_dual_cohort_20261003.py evidence/
cp /workspace/test.csv /workspace/puzzle_info.json evidence/recovered_campaign_inputs_20261001/
nvidia-smi > /workspace/pipeline_hardware_54195079.log
nvidia-smi topo -m >> /workspace/pipeline_hardware_54195079.log
nvcc --version >> /workspace/pipeline_hardware_54195079.log
command -v nsys || true
