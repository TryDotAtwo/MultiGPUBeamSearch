set -euo pipefail
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
out=/workspace/beam-results/native-library/red
mkdir -p "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
set +e
python -m pytest -q integrations/cayleypy_native/tests/test_native_library.py > "$out/pytest.log" 2>&1
rc=$?
set -e
cat "$out/pytest.log"
test "$rc" -ne 0
grep -F "cannot import name 'native_library'" "$out/pytest.log"
