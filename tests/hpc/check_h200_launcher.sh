#!/usr/bin/env bash
set -euo pipefail
root=$(pwd)
tmp=$(mktemp -d)
export H200_REPO_DIR="$root" H200_RUN_ROOT="$tmp" H200_BUILD_DIR="$tmp"
export H200_PYTHON=${H200_PYTHON:-python}
# Identity is simulated here; this script never proves actual H200 execution.
printf '#!/bin/bash\nprintf "NVIDIA H200, 143771, 9.0\\n%%.0s" {1..8}\n' > "$tmp/smi"
chmod +x "$tmp/smi"
export H200_SMI="$tmp/smi"
export H200_DATA_DIR=${H200_TEST_DATA_DIR:?set genuine test data directory}
export BEAM_WEIGHT_DIR=${H200_TEST_WEIGHT_DIR:?set genuine sealed test weight directory}
cp tests/hpc/h200_fake_runner.sh "$tmp/production_runner"
chmod +x "$tmp/production_runner"
bash -n hpc/h200_dual_large_beam.sh hpc/h200_8gpu_256m.sh
bash hpc/h200_8gpu_256m.sh
logs=("$tmp"/h200_run.*/rank*.log)
[[ ${#logs[@]} == 8 ]]
for log in "${logs[@]}"; do grep -q 'mock_pass' "$log"; done
negative=$(mktemp -d)
if H200_RUN_ROOT="$negative" H200_DATA_DIR="$negative/absent" bash hpc/h200_8gpu_256m.sh; then
  echo 'FAIL: launcher started with missing bundle' >&2
  exit 1
fi
bad_logs=("$negative"/h200_run.*/rank*.log)
[[ ! -e ${bad_logs[0]} ]] || { echo 'FAIL: rank spawned before bundle gate' >&2; exit 1; }
wrong_build=$(mktemp -d)
if H200_RUN_ROOT="$wrong_build" H200_TEST_BUILD_ARCH=75 bash hpc/h200_8gpu_256m.sh; then
  echo 'FAIL: launcher started incompatible build' >&2
  exit 1
fi
wrong_logs=("$wrong_build"/h200_run.*/rank*.log)
[[ ! -e ${wrong_logs[0]} ]] || { echo 'FAIL: rank spawned before build gate' >&2; exit 1; }
echo 'PASS: eight ranks, capacity, exchange scale, arguments; mock only, no CUDA'
