#!/usr/bin/env bash
set -euo pipefail
root=/workspace/beam-results/rust-mlp-verify-20261114
mkdir -p "$root" /workspace/tmp /workspace/beam-work /workspace/cutlass
trap 'rc=$?; printf "FAILED rc=%s line=%s\n" "$rc" "$LINENO" >> "$root/status.txt"' ERR
echo SETUP_START >> "$root/status.txt"
python - <<'PY' > "$root/inputs-before.txt"
import hashlib, json, tarfile
from pathlib import Path
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
with tarfile.open('/workspace/verify-overlay.tar.gz', 'r:gz') as tar:
    manifest=json.load(tar.extractfile('manifest.json'))
assert sha('/workspace/verify-base.tar.gz')==manifest['base_sha256']
assert sha('/workspace/verify-cutlass.tar.gz')==manifest['cutlass_sha256']
print('base', manifest['base_commit'], manifest['base_sha256'])
print('cutlass', manifest['cutlass_sha256'])
PY
tar -xzf /workspace/verify-base.tar.gz -C /workspace/beam-work
tar -xzf /workspace/verify-cutlass.tar.gz -C /workspace/cutlass
python - <<'PY' >> "$root/inputs-before.txt"
import hashlib, json, tarfile
from pathlib import Path
work=Path('/workspace/beam-work')
with tarfile.open('/workspace/verify-overlay.tar.gz', 'r:gz') as tar:
    manifest=json.load(tar.extractfile('manifest.json'))
    for name, expected in manifest['baseline'].items():
        assert hashlib.sha256((work/name).read_bytes()).hexdigest()==expected, name
    print('verified baseline files', len(manifest['baseline']))
    for name, row in manifest['files'].items():
        path=work/name
        assert path.resolve().is_relative_to(work)
        data=tar.extractfile(name).read()
        assert hashlib.sha256(data).hexdigest()==row['sha256'], name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(row['sha256'], name)
PY
export TMPDIR=/workspace/tmp CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target
export PATH="/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH" PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TORCH_CUDA_ARCH_LIST=8.6 CARGO_BUILD_JOBS=2
if ! command -v cargo >/dev/null; then
  curl --retry 4 --retry-all-errors --retry-delay 2 --connect-timeout 15 --max-time 120 --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /workspace/tmp/rustup.sh
  sh /workspace/tmp/rustup.sh -y --profile minimal --no-modify-path > "$root/rustup.log" 2>&1
fi
python -m pip install -q pytest 'cayleypy>=0.1.0' 'cmake==4.1.0' ninja > "$root/pip.log" 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$root/gpus.csv"
python -c 'import platform,torch; print(platform.platform()); print(torch.__version__); print(torch.version.cuda); assert torch.cuda.device_count() >= 2; print("GPU_COUNT",torch.cuda.device_count())' > "$root/software.txt"
rustc --version >> "$root/software.txt"
cargo --version >> "$root/software.txt"
nvcc --version >> "$root/software.txt"
cd /workspace/beam-work
echo CARGO_TESTS_START >> "$root/status.txt"
cargo test --workspace --locked > "$root/cargo-tests.log" 2>&1
echo CARGO_TESTS_OK >> "$root/status.txt"
echo BUILD_START >> "$root/status.txt"
cargo build --release --locked -p beam-python -p beam-cli > "$root/cargo-build.log" 2>&1
cp /workspace/target/release/lib_beam_runtime.so integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so
mkdir -p integrations/cayleypy_native/multigpubeamsearch/_bin
cp /workspace/target/release/beam-runner integrations/cayleypy_native/multigpubeamsearch/_bin/beam-runner
sha256sum integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so integrations/cayleypy_native/multigpubeamsearch/_bin/beam-runner > "$root/rust-binaries.sha256"
echo BUILD_OK >> "$root/status.txt"
PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native/multigpubeamsearch python rust/beam-python/tests/mlp_snapshot.py > "$root/pyo3-test.log" 2>&1
echo PYO3_TEST_OK >> "$root/status.txt"
echo PYTHON_TESTS_START >> "$root/status.txt"
python -m pytest -q integrations/cayleypy_native/tests --tb=short --basetemp /workspace/tmp/verify-pytest -p no:cacheprovider > "$root/python-tests.log" 2>&1
echo PYTHON_TESTS_OK >> "$root/status.txt"
echo GPU_TEST_START >> "$root/status.txt"
python tests/public_mlp_launch_verify_gpu.py --source /workspace/beam-work --cutlass /workspace/cutlass --cache "$root/cache" --out "$root/gpu" > "$root/gpu.log" 2>&1
echo GPU_TEST_OK >> "$root/status.txt"
python - <<'PY' > "$root/inputs-after.txt"
import hashlib,json,tarfile
from pathlib import Path
work=Path('/workspace/beam-work')
with tarfile.open('/workspace/verify-overlay.tar.gz','r:gz') as tar:
    manifest=json.load(tar.extractfile('manifest.json'))
for name,expected in sorted(manifest['baseline'].items()):
    expected=manifest['files'].get(name, {}).get('sha256', expected)
    actual=hashlib.sha256((work/name).read_bytes()).hexdigest()
    assert actual==expected,name
    print(actual,name)
for name,row in sorted(manifest['files'].items()):
    if name not in manifest['baseline']:
        actual=hashlib.sha256((work/name).read_bytes()).hexdigest()
        assert actual==row['sha256'],name
        print(actual,name)
PY
echo RUN_COMPLETE >> "$root/status.txt"
