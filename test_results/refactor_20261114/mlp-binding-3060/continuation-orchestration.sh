#!/usr/bin/env bash
set -euo pipefail
root=/workspace/beam-results/mlp-binding-20261114
trap 'rc=$?; printf "CONTINUE_FAILED rc=%s line=%s\n" "$rc" "$LINENO" >> "$root/status.txt"' ERR
export TMPDIR=/workspace/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
export PATH="/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=8.6
printf '%s\n' CONTINUE_WITH_GIT_ARCHIVE_CUTLASS >> "$root/status.txt"
python - <<'PYCODE' >> "$root/source-digest.txt"
from pathlib import Path
from multigpubeamsearch.build import source_digest
actual=source_digest(Path('/workspace/cutlass'),cutlass=True)
print('verified-git-archive-cutlass',actual)
assert actual=='c6a6fa64a544036c6ecd9a9875e6985d456f4186a8aeb9a82e0132c3897a4645'
PYCODE
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
