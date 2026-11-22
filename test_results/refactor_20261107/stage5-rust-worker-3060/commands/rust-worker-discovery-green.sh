set -euo pipefail
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
out=/workspace/beam-results/rust-runner-wheel/discovery-green
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
python -m pytest -q integrations/cayleypy_native/tests/test_rust_control.py > "$out/pytest.log" 2>&1 || { cat "$out/pytest.log"; exit 1; }
cat "$out/pytest.log"
echo WORKER_DISCOVERY_SUCCESS > "$out/success.txt"
