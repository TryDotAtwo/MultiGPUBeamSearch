#!/bin/bash
set -uo pipefail
work=/workspace/beam-work
base=/workspace/beam-results/native-generator
root=/workspace/beam-results/graph-sanitizer-detail
mkdir -p "$root"
export TMPDIR=/workspace/tmp PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$work/integrations/cayleypy_native:$work/integrations/cayleypy_native/multigpubeamsearch:$work/tests"
export PATH="/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6 CARGO_BUILD_JOBS=2
compute-sanitizer --version > "$root/version.txt" 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$root/gpus.csv"
nvidia-smi topo -m > "$root/topology.txt" 2>&1
mapfile -t libraries < <(find "$base/cache/builds" -type f -name libbeam_native.so)
if [[ ${#libraries[@]} != 1 ]]; then echo 'library count mismatch' > "$root/error.txt"; exit 1; fi
cp "$(dirname "${libraries[0]}")/native-library-build.json" "$root/native-library-build.json"
sha256sum /workspace/graph-overlay.tar.gz /workspace/graph-sanitizer-driver.py \
    /workspace/graph-sanitizer-detail-run.sh "$work/integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so" \
    "$work/integrations/cayleypy_native/multigpubeamsearch/_bin/beam-runner" "${libraries[0]}" > "$root/input-sha256.txt"
printf '%s\n' 'timeout 420 compute-sanitizer --print-session-details --target-processes all --tool memcheck --leak-check full --error-exitcode 99 --log-file memcheck-%p.log python graph-sanitizer-driver.py' > "$root/command.txt"
timeout 420 compute-sanitizer --print-session-details --target-processes all --tool memcheck --leak-check full --error-exitcode 99 \
    --log-file "$root/memcheck-%p.log" python /workspace/graph-sanitizer-driver.py \
    --source "$work" --cutlass /workspace/cutlass --cache "$base/cache" --out "$root/search" > "$root/driver.log" 2>&1
result=$?
printf '%s\n' "$result" > "$root/exit.txt"
tar -C /workspace -czf /workspace/graph-sanitizer-detail-evidence.tar.gz beam-results/graph-sanitizer-detail graph-sanitizer-driver.py graph-sanitizer-detail-run.sh
sha256sum /workspace/graph-sanitizer-detail-evidence.tar.gz > /workspace/graph-sanitizer-detail-evidence.sha256
echo SANITIZER_DETAIL_ARCHIVE_READY
exit "$result"
