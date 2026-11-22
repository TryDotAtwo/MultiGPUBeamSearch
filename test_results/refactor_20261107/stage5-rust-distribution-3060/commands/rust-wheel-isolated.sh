set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp CARGO_NET_OFFLINE=true
out=/workspace/beam-results/rust-wheel/isolated
mkdir -p "$out/wheel"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
python -m pip wheel --no-deps --wheel-dir "$out/wheel" ./integrations/cayleypy_native > "$out/pip.log" 2>&1 || { tail -n 100 "$out/pip.log"; exit 1; }
sha256sum "$out"/wheel/*.whl > "$out/wheel.sha256"
python - "$out"/wheel/*.whl <<'PY' > "$out/content.log"
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as wheel:
    names=set(wheel.namelist())
    for name in ('multigpubeamsearch/_beam_runtime.abi3.so', 'multigpubeamsearch/_native_source/source_manifest.json', 'multigpubeamsearch/_native_source/cuda/cnot_domain.cu', 'cayleypy_native/__init__.py'):
        assert name in names, name
    print('ISOLATED_CONTENT_OK',len(names))
PY
cat "$out/content.log"
tail -n 8 "$out/pip.log"
echo RUST_WHEEL_ISOLATED_SUCCESS > "$out/success.txt"
