set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
out=/workspace/beam-results/native-library/reconfigure-green
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
python -m pytest -q integrations/cayleypy_native/tests/test_native_library.py integrations/cayleypy_native/tests/test_build.py integrations/cayleypy_native/tests/test_rust_control.py > "$out/pytest.log" 2>&1 || { cat "$out/pytest.log"; exit 1; }
cat "$out/pytest.log"
echo NATIVE_LIBRARY_RECONFIGURE_TESTS_SUCCESS > "$out/success.txt"
