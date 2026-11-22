#!/bin/bash
set -Eeuo pipefail
root=/workspace/beam-results/rust-graph-contract
mkdir -p "$root" /workspace/tmp /workspace/beam-work /workspace/cutlass
echo SOURCE_VERIFY_START >> "$root/status.txt"
python - <<'PY' > "$root/source-before.txt"
import hashlib,json,tarfile
from pathlib import Path
sha=lambda b:hashlib.sha256(b).hexdigest()
with tarfile.open('/workspace/graph-overlay.tar.gz','r:gz') as tar:
    manifest=json.load(tar.extractfile('manifest.json'))
assert sha(Path('/workspace/graph-base.tar.gz').read_bytes())==manifest['base_sha256']
assert sha(Path('/workspace/graph-cutlass.tar.gz').read_bytes())==manifest['cutlass_sha256']
print('base',manifest['base_commit'],manifest['base_sha256'])
print('cutlass',manifest['cutlass_sha256'])
PY
tar -xzf /workspace/graph-base.tar.gz -C /workspace/beam-work
tar -xzf /workspace/graph-cutlass.tar.gz -C /workspace/cutlass
tar -xzf /workspace/graph-overlay.tar.gz -C /workspace/beam-work --exclude=manifest.json
python - <<'PY' >> "$root/source-before.txt"
import hashlib,json,tarfile
from pathlib import Path
work=Path('/workspace/beam-work')
with tarfile.open('/workspace/graph-overlay.tar.gz','r:gz') as tar:
    manifest=json.load(tar.extractfile('manifest.json'))
expected=dict(manifest['baseline'])
expected.update({name:row['sha256'] for name,row in manifest['files'].items()})
for name,digest in sorted(expected.items()):
    actual=hashlib.sha256((work/name).read_bytes()).hexdigest()
    assert actual==digest,(name,actual,digest)
print('verified source files',len(expected))
PY
echo SOURCE_VERIFY_OK >> "$root/status.txt"
export TMPDIR=/workspace/tmp CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target
export PATH="/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native:/workspace/beam-work/integrations/cayleypy_native/multigpubeamsearch
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TORCH_CUDA_ARCH_LIST=8.6 CARGO_BUILD_JOBS=2
if ! command -v cargo >/dev/null; then
    curl --retry 4 --retry-all-errors --retry-delay 2 --connect-timeout 15 --max-time 120 --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /workspace/tmp/rustup.sh
    sh /workspace/tmp/rustup.sh -y --profile minimal --no-modify-path > "$root/rustup.log" 2>&1
fi
python -m pip install -q pytest 'cayleypy>=0.1.0' 'cmake==4.1.0' ninja > "$root/pip.log" 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$root/gpus.csv"
python -c 'import platform,torch;print(platform.platform());print(torch.__version__);print(torch.version.cuda);assert torch.cuda.device_count()>=2;print("GPU_COUNT",torch.cuda.device_count())' > "$root/software.txt"
rustc --version >> "$root/software.txt"
cargo --version >> "$root/software.txt"
nvcc --version >> "$root/software.txt"
cd /workspace/beam-work
echo CARGO_TEST_START >> "$root/status.txt"
cargo test --workspace --locked > "$root/cargo-tests.log" 2>&1
echo CARGO_TEST_OK >> "$root/status.txt"
echo BUILD_START >> "$root/status.txt"
cargo build --release --locked -p beam-python -p beam-cli > "$root/cargo-build.log" 2>&1
cp /workspace/target/release/lib_beam_runtime.so integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so
mkdir -p integrations/cayleypy_native/multigpubeamsearch/_bin
cp /workspace/target/release/beam-runner integrations/cayleypy_native/multigpubeamsearch/_bin/beam-runner
sha256sum integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so integrations/cayleypy_native/multigpubeamsearch/_bin/beam-runner > "$root/rust-binaries.sha256"
echo BUILD_OK >> "$root/status.txt"
python rust/beam-python/tests/search_session.py integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so > "$root/pyo3-session.log" 2>&1
python rust/beam-python/tests/interrupt.py integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so > "$root/pyo3-interrupt.log" 2>&1
echo PYO3_OK >> "$root/status.txt"
echo PYTHON_START >> "$root/status.txt"
python -m pytest -q integrations/cayleypy_native/tests --tb=short --basetemp /workspace/tmp/graph-pytest -p no:cacheprovider > "$root/python-tests.log" 2>&1
echo PYTHON_OK >> "$root/status.txt"
echo GPU_START >> "$root/status.txt"
python tests/public_rust_graph_gpu.py --source /workspace/beam-work --cutlass /workspace/cutlass --cache "$root/cache" --out "$root/gpu" > "$root/gpu.log" 2>&1
echo GPU_OK >> "$root/status.txt"
python - <<'PY' > "$root/source-after.txt"
import hashlib,json,tarfile
from pathlib import Path
work=Path('/workspace/beam-work')
with tarfile.open('/workspace/graph-overlay.tar.gz','r:gz') as tar:
    manifest=json.load(tar.extractfile('manifest.json'))
expected=dict(manifest['baseline'])
expected.update({name:row['sha256'] for name,row in manifest['files'].items()})
for name,digest in sorted(expected.items()):
    actual=hashlib.sha256((work/name).read_bytes()).hexdigest()
    assert actual==digest,(name,actual,digest)
print('verified source files',len(expected))
PY
echo COMPLETE >> "$root/status.txt"
