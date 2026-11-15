#!/bin/bash
set -Eeuo pipefail
out=/workspace/conditional-diagnostic-v2-results
source_root=/workspace/finite-v2-work
mkdir -p "$out/source/cuda" "$out/source/src"
finish() {
    code=$?
    trap - EXIT
    printf 'EXIT_CODE=%s\n' "$code" >> "$out/status.txt"
    tar -czf /workspace/conditional-diagnostic-v2-evidence.tar.gz -C /workspace conditional-diagnostic-v2-results
    sha256sum /workspace/conditional-diagnostic-v2-evidence.tar.gz > /workspace/conditional-diagnostic-v2-evidence.sha256
    exit "$code"
}
trap finish EXIT
printf '%s  /workspace/conditional-diagnostic.cu\n' "$1" | sha256sum -c -
cp /workspace/conditional-diagnostic.cu "$out/source/"
cp "$source_root/cuda/radix128.cuh" "$out/source/cuda/"
cp "$source_root/src/types.hpp" "$source_root/src/config.hpp" "$out/source/src/"
sha256sum "$out"/source/conditional-diagnostic.cu "$out"/source/cuda/radix128.cuh "$out"/source/src/* > "$out/source-sha256.txt"
{ /usr/local/cuda/bin/nvcc --version; compute-sanitizer --version; nvidia-smi --query-gpu=name,driver_version --format=csv; } > "$out/software.txt"
echo COMPILE_START >> "$out/status.txt"
/usr/local/cuda/bin/nvcc -O3 -lineinfo -std=c++17 -arch=sm_86 -I"$source_root" \
    -DBEAM_STATE_LOGICAL_BYTES=96 -DBEAM_STATE_PHYSICAL_BYTES=112 -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=24 \
    /workspace/conditional-diagnostic.cu -o /workspace/conditional-diagnostic > "$out/compile.log" 2>&1
sha256sum /workspace/conditional-diagnostic > "$out/binary-sha256.txt"
echo COMPILE_OK >> "$out/status.txt"
for n in 1 1025; do
  for mode in plain conditional; do
    echo "START $mode $n" >> "$out/status.txt"
    set +e
    timeout 90 /workspace/conditional-diagnostic "$mode" "$n" > "$out/$mode-$n-direct.log" 2>&1
    code=$?
    set -e
    printf '%s\n' "$code" > "$out/$mode-$n-direct.exit"
    for tool in memcheck racecheck synccheck; do
      set +e
      timeout 90 compute-sanitizer --tool "$tool" --error-exitcode=86 \
        /workspace/conditional-diagnostic "$mode" "$n" > "$out/$mode-$n-$tool.log" 2>&1
      code=$?
      set -e
      printf '%s\n' "$code" > "$out/$mode-$n-$tool.exit"
    done
    echo "DONE $mode $n" >> "$out/status.txt"
  done
done
echo COMPLETE >> "$out/status.txt"
