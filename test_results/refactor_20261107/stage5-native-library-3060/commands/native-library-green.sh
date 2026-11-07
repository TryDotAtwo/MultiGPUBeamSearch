set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
out=/workspace/beam-results/native-library/green
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
python -m pytest -q integrations/cayleypy_native/tests/test_native_library.py integrations/cayleypy_native/tests/test_build.py integrations/cayleypy_native/tests/test_rust_control.py > "$out/pytest.log" 2>&1 || { cat "$out/pytest.log"; exit 1; }
cargo test --locked --offline -p beam-native-sys -p beam-cli > "$out/cargo-test.log" 2>&1 || { tail -n 100 "$out/cargo-test.log"; exit 1; }
cargo build --release --locked --offline -p beam-cli > "$out/cargo-build.log" 2>&1 || { tail -n 100 "$out/cargo-build.log"; exit 1; }
for kind in beam_native beam_native_libtorch; do
  python tests/native_abi_probe.py --runner /workspace/target/release/beam-runner \
    --library "/workspace/beam-results/rust-wheel/gpu/build/lib${kind}.so" \
    --state-len 96 --storage-len 112 --move-count 24 --out "$out/$kind.json" >> "$out/probe.log" 2>&1 || { cat "$out/probe.log"; exit 1; }
done
cat "$out/pytest.log"
cat "$out/probe.log"
echo NATIVE_LIBRARY_GREEN_SUCCESS > "$out/success.txt"
