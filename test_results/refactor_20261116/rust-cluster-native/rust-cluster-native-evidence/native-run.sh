#!/bin/bash
set -Eeuo pipefail
out=/workspace/rust-cluster-native-evidence
mkdir -p "$out" /workspace/cutlass
finish() {
  code=$?
  trap - EXIT
  printf 'EXIT_CODE=%s\n' "$code" >> "$out/status.txt"
  cp /workspace/cluster_native_rank_gpu.py /workspace/native-run.sh "$out/"
  tar -czf /workspace/rust-cluster-native-evidence.tar.gz --exclude='*/cache' -C /workspace rust-cluster-native-evidence
  sha256sum /workspace/rust-cluster-native-evidence.tar.gz > /workspace/rust-cluster-native-evidence.sha256
  exit "$code"
}
trap finish EXIT
printf '%s  %s\n' "$1" /workspace/cutlass-source.tar.gz | sha256sum -c - > "$out/inputs.txt"
printf '%s  %s\n' "$2" /workspace/cluster_native_rank_gpu.py | sha256sum -c - >> "$out/inputs.txt"
tar -xzf /workspace/cutlass-source.tar.gz -C /workspace/cutlass
export TMPDIR=/workspace/tmp CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target
export PATH="$CARGO_HOME/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH=/workspace/rust-cluster-src/integrations/cayleypy_native:/workspace/rust-cluster-src/tests
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TORCH_CUDA_ARCH_LIST=8.6
cd /workspace/rust-cluster-src
echo NATIVE_SEARCH >> "$out/status.txt"
timeout 1500 python /workspace/cluster_native_rank_gpu.py --source /workspace/rust-cluster-src --cutlass /workspace/cutlass --out "$out/search" > "$out/native-search.log" 2>&1
echo COMPLETE >> "$out/status.txt"
