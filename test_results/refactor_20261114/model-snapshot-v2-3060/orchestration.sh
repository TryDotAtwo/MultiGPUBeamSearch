#!/usr/bin/env bash
set -euo pipefail
expected_commit="$1"
root=/workspace/beam-results/model-snapshot-v2-20261114
cache=/workspace/beam-results/public-index-20261114/cache-cutlass
mkdir -p "$root"
trap 'rc=$?; printf "FAILED rc=%s line=%s\n" "$rc" "$LINENO" >> "$root/status.txt"' ERR
export TMPDIR=/workspace/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
export PATH="/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=8.6
cd /workspace/beam-work
echo UPDATE_START >> "$root/status.txt"
git fetch --depth=5 origin main
git merge --ff-only FETCH_HEAD
test "$(git rev-parse HEAD)" = "$expected_commit"
git rev-parse HEAD > "$root/commit.txt"
git -C /workspace/cutlass rev-parse HEAD > "$root/cutlass-commit.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$root/gpus.csv"
python -c 'import platform,torch; print(platform.platform()); print(torch.__version__); print(torch.version.cuda)' > "$root/software.txt"
nvcc --version >> "$root/software.txt"
python - <<'PY' > "$root/source-digest.txt"
from pathlib import Path
from multigpubeamsearch.build import source_digest
native = source_digest(Path('/workspace/beam-work'))
cutlass = source_digest(Path('/workspace/cutlass'), cutlass=True)
print('native', native)
print('cutlass', cutlass)
assert native == '91c6a96c34a894f97d84964cf25f4f98ae4f4618d2be9f3063d30648fff37557'
assert cutlass == '0adcc47dbf4082dc2acbf0bd07545aace5a63dcab8926f919a2e4b236d314e56'
PY
sha256sum integrations/cayleypy_native/multigpubeamsearch/{model_snapshot.py,persistent.py,dispatch.py,_beam_runtime.abi3.so,_bin/beam-runner} \
 tests/public_model_snapshot_gpu.py > "$root/inputs-before.sha256"
echo CPU_TESTS_START >> "$root/status.txt"
python -m pytest -q integrations/cayleypy_native/tests --basetemp /workspace/tmp/snapshot-v2-pytest -p no:cacheprovider > "$root/cpu-tests.log" 2>&1
echo CPU_TESTS_OK >> "$root/status.txt"
echo GPU_TEST_START >> "$root/status.txt"
python tests/public_model_snapshot_gpu.py --source /workspace/beam-work --cutlass /workspace/cutlass \
 --cache "$cache" --out "$root/gpu" > "$root/gpu.log" 2>&1
echo GPU_TEST_OK >> "$root/status.txt"
sha256sum integrations/cayleypy_native/multigpubeamsearch/{model_snapshot.py,persistent.py,dispatch.py,_beam_runtime.abi3.so,_bin/beam-runner} \
 tests/public_model_snapshot_gpu.py > "$root/inputs-after.sha256"
cmp "$root/inputs-before.sha256" "$root/inputs-after.sha256"
echo RUN_COMPLETE >> "$root/status.txt"
