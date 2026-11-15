#!/bin/bash
set -Eeuo pipefail
out=/workspace/beam-results/cnot-rust-validation-final
work=/workspace/beam-work
mkdir -p "$out" "$work" /workspace/tmp
finish() {
  code=$?
  trap - EXIT
  printf 'EXIT_CODE=%s\n' "$code" >> "$out/status.txt"
  tar -czf /workspace/cnot-rust-evidence-final.tar.gz -C /workspace/beam-results cnot-rust-validation-final
  sha256sum /workspace/cnot-rust-evidence-final.tar.gz > /workspace/cnot-rust-evidence-final.sha256
  exit "$code"
}
trap finish EXIT
tar -xzf /workspace/cnot-base.tar.gz -C "$work"
tar -xzf /workspace/cnot-overlay.tar.gz -C "$work" --exclude=manifest.json
tar -xOf /workspace/cnot-overlay.tar.gz manifest.json > "$out/source-manifest.json"
verify_source() {
python - "$work" "$out/source-manifest.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); m=json.loads(Path(sys.argv[2]).read_text())
expected=dict(m['baseline'], **{name: item['sha256'] for name,item in m['overlays'].items()})
for name, digest in sorted(expected.items()):
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert actual==digest,(name,actual,digest)
print('verified source files',len(expected),'base',m['base_commit'])
PY
}
verify_source > "$out/source-before.txt"
echo SOURCE_OK >> "$out/status.txt"
export TMPDIR=/workspace/tmp CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target
export PATH="/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$work/integrations/cayleypy_native:$work/tests"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TORCH_CUDA_ARCH_LIST=8.6 CARGO_BUILD_JOBS=2 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/workspace/torch-extensions
if ! command -v cargo >/dev/null; then
  curl --retry 4 --retry-all-errors --retry-delay 2 --connect-timeout 15 --max-time 120 --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /workspace/tmp/rustup.sh
  sh /workspace/tmp/rustup.sh -y --profile minimal --no-modify-path > "$out/rustup.log" 2>&1
fi
python -m pip install -q pytest 'cayleypy>=0.1.0' 'cmake==4.1.0' ninja > "$out/pip.log" 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$out/gpus.csv"
python -c 'import platform,torch;print(platform.platform());print(torch.__version__);print(torch.version.cuda);print("GPU_COUNT",torch.cuda.device_count());assert torch.cuda.device_count()>=2' > "$out/software.txt"
{ rustc --version; cargo --version; nvcc --version; g++ --version | head -n 1; } >> "$out/software.txt"
cd "$work"
echo RUST_START >> "$out/status.txt"
cargo test --workspace --locked > "$out/cargo-tests.log" 2>&1
cargo build --release --locked -p beam-python -p beam-cli > "$out/cargo-build.log" 2>&1
module="$work/integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so"
cp /workspace/target/release/lib_beam_runtime.so "$module"
mkdir -p "$work/integrations/cayleypy_native/multigpubeamsearch/_bin"
cp /workspace/target/release/beam-runner "$work/integrations/cayleypy_native/multigpubeamsearch/_bin/beam-runner"
sha256sum "$module" /workspace/target/release/beam-runner > "$out/binary-sha256.txt"
python rust/beam-python/tests/cnot.py "$module" > "$out/pyo3-contract.log" 2>&1
echo RUST_OK >> "$out/status.txt"
echo PYTHON_START >> "$out/status.txt"
python -m pytest -q integrations/cayleypy_native/tests --tb=short --basetemp /workspace/tmp/cnot-pytest -p no:cacheprovider > "$out/python-tests.log" 2>&1
echo PYTHON_OK >> "$out/status.txt"
echo PUBLIC_CPU_START >> "$out/status.txt"
env CNOT_EVIDENCE_DIR="$out/receipts/cpu-1" python rust/beam-python/tests/cnot_search.py "$module" > "$out/cpu-1.log" 2>&1
timeout 300 env CNOT_EVIDENCE_DIR="$out/receipts/cpu-2" torchrun --standalone --nproc-per-node=2 rust/beam-python/tests/cnot_search.py "$module" > "$out/cpu-2.log" 2>&1
echo PUBLIC_CPU_OK >> "$out/status.txt"
echo PUBLIC_GPU_START >> "$out/status.txt"
timeout 600 env CNOT_EVIDENCE_DIR="$out/receipts/gpu-1" python rust/beam-python/tests/cnot_search.py "$module" --cuda > "$out/gpu-1.log" 2>&1
timeout 600 env CNOT_EVIDENCE_DIR="$out/receipts/gpu-2" torchrun --standalone --nproc-per-node=2 rust/beam-python/tests/cnot_search.py "$module" --cuda > "$out/gpu-2.log" 2>&1
echo PUBLIC_GPU_OK >> "$out/status.txt"
verify_source > "$out/source-after.txt"
echo COMPLETE >> "$out/status.txt"
