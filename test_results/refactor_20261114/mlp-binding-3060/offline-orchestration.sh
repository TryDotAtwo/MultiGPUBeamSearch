#!/usr/bin/env bash
set -euo pipefail
root=/workspace/beam-results/mlp-binding-20261114
mkdir -p "$root" /workspace/tmp
trap 'rc=$?; printf "OFFLINE_FAILED rc=%s line=%s\n" "$rc" "$LINENO" >> "$root/status.txt"' ERR
echo OFFLINE_BOOTSTRAP_START >> "$root/status.txt"
printf '%s  %s\n' 6a0378f1148f08735cc248792758c283784f93e6f890441acd93d737ee708f2c /workspace/mlp-binding-base.tar.gz | sha256sum -c -
printf '%s  %s\n' 458b5d80f344652d61fe011b557968cd58e9583a96f0ab0823ecc4dc9dc0f0c6 /workspace/mlp-binding-cutlass.tar.gz | sha256sum -c -
test ! -e /workspace/beam-work-incomplete-git
if test -e /workspace/beam-work; then mv /workspace/beam-work /workspace/beam-work-incomplete-git; fi
mkdir /workspace/beam-work /workspace/cutlass
tar -xzf /workspace/mlp-binding-base.tar.gz -C /workspace/beam-work
tar -xzf /workspace/mlp-binding-cutlass.tar.gz -C /workspace/cutlass
python /workspace/mlp-binding-restore.py > "$root/overlay.log" 2>&1
printf '%s\n' 65a1952c265bd60c6d69a7ee90687c0a7d336888 > "$root/base-commit.txt"
printf '%s\n' afa1772203677c5118fcd82537a9c8fefbcc7008 > "$root/cutlass-commit.txt"
export TMPDIR=/workspace/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
export PATH="/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=8.6
python -m pip install -q pytest 'cayleypy>=0.1.0' 'cmake==4.1.0' ninja
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$root/gpus.csv"
python -c 'import platform,torch; print(platform.platform()); print(torch.__version__); print(torch.version.cuda); assert torch.cuda.device_count() >= 2; print("GPU_COUNT",torch.cuda.device_count())' > "$root/software.txt"
nvcc --version >> "$root/software.txt"
python - <<'PY' > "$root/source-digest.txt"
from pathlib import Path
from multigpubeamsearch.build import source_digest
print('native',source_digest(Path('/workspace/beam-work')))
cutlass=source_digest(Path('/workspace/cutlass'),cutlass=True)
print('cutlass',cutlass)
assert cutlass=='0adcc47dbf4082dc2acbf0bd07545aace5a63dcab8926f919a2e4b236d314e56'
PY
echo CPU_CTEST_START >> "$root/status.txt"
cmake -S /workspace/beam-work/tests/mlp_binding_host -B /workspace/beam-build/mlp-binding-host > "$root/ctest-configure.log" 2>&1
cmake --build /workspace/beam-build/mlp-binding-host --parallel 2 > "$root/ctest-build.log" 2>&1
ctest --test-dir /workspace/beam-build/mlp-binding-host -R '^mlp_artifact_binding_cpu$' --output-on-failure > "$root/ctest.log" 2>&1
echo CPU_CTEST_OK >> "$root/status.txt"
cd /workspace/beam-work
echo PYTHON_TESTS_START >> "$root/status.txt"
python -m pytest -q integrations/cayleypy_native/tests --basetemp /workspace/tmp/mlp-binding-pytest -p no:cacheprovider > "$root/python-tests.log" 2>&1
echo PYTHON_TESTS_OK >> "$root/status.txt"
echo GPU_TEST_START >> "$root/status.txt"
python tests/public_model_snapshot_gpu.py --source /workspace/beam-work --cutlass /workspace/cutlass \
 --cache "$root/cache" --out "$root/gpu" > "$root/gpu.log" 2>&1
echo GPU_TEST_OK >> "$root/status.txt"
python - <<'PY' > "$root/inputs-after.txt"
import hashlib,json
from pathlib import Path
import tarfile
with tarfile.open('/workspace/mlp-binding-overlay.tar.gz','r:gz') as tar:
 rows=json.load(tar.extractfile('manifest.json'))['files']
for name,row in sorted(rows.items()):
 path=Path('/workspace/beam-work')/name
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 assert digest==row['sha256'],name
 print(digest,name)
PY
echo RUN_COMPLETE >> "$root/status.txt"
