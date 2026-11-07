set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp
export PYTHONPATH=/workspace/beam-results/rust-wheel/final/check/checkout/installed
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.6
out=/workspace/beam-results/rust-wheel/cnot
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
python -m pytest -q -rs integrations/cayleypy_native/tests/test_cnot_native.py > "$out/pytest.log" 2>&1 || { tail -n 100 "$out/pytest.log"; exit 1; }
cat "$out/pytest.log"
echo RUST_WHEEL_CNOT_SUCCESS > "$out/success.txt"
