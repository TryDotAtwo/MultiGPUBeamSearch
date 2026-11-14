#!/usr/bin/env bash
set -euo pipefail
expected_commit="$1"
root=/workspace/beam-results/model-snapshot-20261114
cache=/workspace/beam-results/public-index-20261114/cache-cutlass
mkdir -p "$root" /workspace/tmp
trap 'rc=$?; printf "FAILED rc=%s line=%s\n" "$rc" "$LINENO" >> "$root/status.txt"' ERR
echo BOOTSTRAP_START >> "$root/status.txt"
git clone --filter=blob:none --sparse --depth=1 --branch main https://github.com/TryDotAtwo/MultiGPUBeamSearch.git /workspace/beam-work
git -C /workspace/beam-work sparse-checkout set integrations/cayleypy_native cuda src tools third_party rust native cmake tests kaggle notebooks profiles
git -C /workspace/beam-work checkout --ignore-skip-worktree-bits HEAD -- CMakeLists.txt Cargo.toml Cargo.lock
test "$(git -C /workspace/beam-work rev-parse HEAD)" = "$expected_commit"
git -C /workspace/beam-work rev-parse HEAD > "$root/commit.txt"
git clone --depth=1 --branch v3.8.0 https://github.com/NVIDIA/cutlass.git /workspace/cutlass
git -C /workspace/cutlass rev-parse HEAD > "$root/cutlass-commit.txt"
export TMPDIR=/workspace/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
export PATH="/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=8.6
python -m pip install -q pytest 'cayleypy>=0.1.0' 'cmake==4.1.0' ninja
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$root/gpus.csv"
python -c 'import platform,torch; print(platform.platform()); print(torch.__version__); print(torch.version.cuda)' > "$root/software.txt"
nvcc --version >> "$root/software.txt"
echo GPU_PROBE_START >> "$root/status.txt"
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 2; x=[torch.zeros(1,device=f"cuda:{i}") for i in range(2)]; torch.cuda.synchronize(); print("CUDA_READY")' > "$root/gpu-probe.log" 2>&1
echo GPU_PROBE_OK >> "$root/status.txt"
echo RESTORE_START >> "$root/status.txt"
python /workspace/model-snapshot-restore.py > "$root/restore.log" 2>&1
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
echo RESTORE_OK >> "$root/status.txt"
cd /workspace/beam-work
sha256sum integrations/cayleypy_native/multigpubeamsearch/{model_snapshot.py,persistent.py,dispatch.py,_beam_runtime.abi3.so,_bin/beam-runner} \
 tests/public_model_snapshot_gpu.py > "$root/inputs-before.sha256"
echo CPU_TESTS_START >> "$root/status.txt"
python -m pytest -q integrations/cayleypy_native/tests --basetemp /workspace/tmp/snapshot-pytest -p no:cacheprovider > "$root/cpu-tests.log" 2>&1
echo CPU_TESTS_OK >> "$root/status.txt"
echo GPU_TEST_START >> "$root/status.txt"
python tests/public_model_snapshot_gpu.py --source /workspace/beam-work --cutlass /workspace/cutlass \
 --cache "$cache" --out "$root/gpu" > "$root/gpu.log" 2>&1
echo GPU_TEST_OK >> "$root/status.txt"
sha256sum integrations/cayleypy_native/multigpubeamsearch/{model_snapshot.py,persistent.py,dispatch.py,_beam_runtime.abi3.so,_bin/beam-runner} \
 tests/public_model_snapshot_gpu.py > "$root/inputs-after.sha256"
cmp "$root/inputs-before.sha256" "$root/inputs-after.sha256"
echo RUN_COMPLETE >> "$root/status.txt"
