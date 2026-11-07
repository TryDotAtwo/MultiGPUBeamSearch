set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp
export PYTHONPATH=/workspace/beam-results/rust-wheel/final/check/checkout/installed
out=/workspace/beam-results/rust-wheel/pytest
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
python -m pytest -q integrations/cayleypy_native/tests > "$out/pytest.log" 2>&1 || { tail -n 100 "$out/pytest.log"; exit 1; }
tail -n 20 "$out/pytest.log"
echo RUST_WHEEL_PYTEST_SUCCESS > "$out/success.txt"
