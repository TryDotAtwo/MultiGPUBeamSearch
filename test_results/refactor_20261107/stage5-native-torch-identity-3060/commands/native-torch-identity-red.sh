set -euo pipefail
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
out=/workspace/beam-results/native-torch-identity/red
mkdir -p "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
set +e
python -m pytest -q integrations/cayleypy_native/tests/test_native_library.py -k torch_dependent_runner > "$out/pytest.log" 2>&1
rc=$?
set -e
cat "$out/pytest.log"
test "$rc" -ne 0
grep -F 'DID NOT RAISE' "$out/pytest.log"
echo EXPECTED_MISSING_TORCH_IDENTITY_FAILURE > "$out/success.txt"
