set -euo pipefail
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
out=/workspace/beam-results/rust-runner-wheel/discovery-red
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
set +e
python -m pytest -q integrations/cayleypy_native/tests/test_rust_control.py::test_packaged_worker_discovery_requires_an_executable > "$out/pytest.log" 2>&1
rc=$?
set -e
cat "$out/pytest.log"
test "$rc" -ne 0
grep -F "has no attribute 'worker_executable'" "$out/pytest.log"
