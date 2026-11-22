#!/bin/bash
set -Eeuo pipefail
root=/workspace/beam-results/native-generator
base=/workspace/beam-base
work=/workspace/beam-work
mkdir -p "$root" /workspace/tmp "$base" "$work" /workspace/cutlass
python - <<'PY' > "$root/source-before.txt"
import hashlib,json,tarfile
from pathlib import Path
sha=lambda b:hashlib.sha256(b).hexdigest()
with tarfile.open('/workspace/graph-overlay.tar.gz','r:gz') as tar:
    m=json.load(tar.extractfile('manifest.json'))
assert sha(Path('/workspace/graph-base.tar.gz').read_bytes())==m['base_sha256']
assert sha(Path('/workspace/graph-cutlass.tar.gz').read_bytes())==m['cutlass_sha256']
print('base',m['base_commit'],m['base_sha256'])
print('cutlass',m['cutlass_sha256'])
PY
tar -xzf /workspace/graph-base.tar.gz -C "$base"
tar -xzf /workspace/graph-base.tar.gz -C "$work"
tar -xzf /workspace/graph-cutlass.tar.gz -C /workspace/cutlass
tar -xzf /workspace/graph-overlay.tar.gz -C "$work" --exclude=manifest.json
verify_source() {
python - <<'PY'
import hashlib,json,tarfile
from pathlib import Path
with tarfile.open('/workspace/graph-overlay.tar.gz','r:gz') as tar:m=json.load(tar.extractfile('manifest.json'))
for root,expected in ((Path('/workspace/beam-base'),m['baseline']),
    (Path('/workspace/beam-work'),dict(m['baseline'],**{k:v['sha256'] for k,v in m['files'].items()}))):
    for name,digest in sorted(expected.items()):
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        assert actual==digest,(str(root),name,actual,digest)
    print(root,'verified source files',len(expected))
PY
}
verify_source >> "$root/source-before.txt"
echo SOURCE_VERIFY_OK >> "$root/status.txt"
export TMPDIR=/workspace/tmp CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target
export PATH="/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$work/integrations/cayleypy_native:$work/integrations/cayleypy_native/multigpubeamsearch:$work/tests"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TORCH_CUDA_ARCH_LIST=8.6 CARGO_BUILD_JOBS=2
if ! command -v cargo >/dev/null; then
    curl --retry 4 --retry-all-errors --retry-delay 2 --connect-timeout 15 --max-time 120 --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /workspace/tmp/rustup.sh
    sh /workspace/tmp/rustup.sh -y --profile minimal --no-modify-path > "$root/rustup.log" 2>&1
fi
python -m pip install -q pytest 'cayleypy>=0.1.0' 'cmake==4.1.0' ninja > "$root/pip.log" 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$root/gpus.csv"
python -c 'import platform,torch;print(platform.platform());print(torch.__version__);print(torch.version.cuda);assert torch.cuda.device_count()>=2;print("GPU_COUNT",torch.cuda.device_count())' > "$root/software.txt"
{ rustc --version; cargo --version; nvcc --version; g++ --version | head -n 1; } >> "$root/software.txt"
echo CPU_TEST_START >> "$root/status.txt"
for shape in 120-128-18 150-160-255; do
    IFS=- read -r logical physical count <<< "$shape"
    defs=(-DBEAM_STATE_LOGICAL_BYTES="$logical" -DBEAM_STATE_PHYSICAL_BYTES="$physical" -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT="$count")
    g++ -std=c++20 -O2 "${defs[@]}" "$work/tests/native_loaded_graph_hash_tests.cpp" "$work/src/native_input.cpp" -o "/workspace/tmp/hash-$shape" >> "$root/cpu-build.log" 2>&1
    "/workspace/tmp/hash-$shape" >> "$root/hash-tests.log" 2>&1
done
defs=(-DBEAM_STATE_LOGICAL_BYTES=120 -DBEAM_STATE_PHYSICAL_BYTES=128 -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=18)
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer "${defs[@]}" "$work/tests/native_loaded_graph_hash_tests.cpp" "$work/src/native_input.cpp" -o /workspace/tmp/hash-asan >> "$root/cpu-build.log" 2>&1
ASAN_OPTIONS=detect_leaks=1 /workspace/tmp/hash-asan > "$root/hash-asan.log" 2>&1
g++ -std=c++20 -O1 -pthread -I "$work/native" "$work/native/beam_native.cpp" "$work/tests/native_abi_tests.cpp" -o /workspace/tmp/abi-tests >> "$root/cpu-build.log" 2>&1
/workspace/tmp/abi-tests > "$root/abi-tests.log" 2>&1
g++ -std=c++20 -O1 -fPIC -shared -pthread -I "$work/native" "$work/native/beam_native.cpp" "$work/tests/native_abi_tests.cpp" -o /workspace/tmp/beam-fixture.so >> "$root/cpu-build.log" 2>&1
g++ -std=c++20 -O1 -fPIC -shared -pthread -I "$base/native" "$base/native/beam_native.cpp" "$base/tests/native_abi_tests.cpp" -o /workspace/tmp/beam-fixture-legacy.so >> "$root/cpu-build.log" 2>&1
echo CPU_TEST_OK >> "$root/status.txt"
cd "$work"
echo CARGO_TEST_START >> "$root/status.txt"
cargo test --workspace --locked > "$root/cargo-tests.log" 2>&1
BEAM_TEST_LIBRARY=/workspace/tmp/beam-fixture.so BEAM_TEST_LEGACY_LIBRARY=/workspace/tmp/beam-fixture-legacy.so \
    cargo test --locked -p beam-native-sys --test context -- --ignored > "$root/ffi-tests.log" 2>&1
cargo build --release --locked -p beam-python -p beam-cli > "$root/cargo-build.log" 2>&1
cp /workspace/target/release/lib_beam_runtime.so integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so
mkdir -p integrations/cayleypy_native/multigpubeamsearch/_bin
cp /workspace/target/release/beam-runner integrations/cayleypy_native/multigpubeamsearch/_bin/beam-runner
worker=/workspace/target/release/beam-runner
"$worker" --resident-protocol-info > "$root/protocol-legacy.json"
BEAM_REQUIRE_LOADED_GRAPH_BINDING=1 "$worker" --resident-protocol-info > "$root/protocol-binding.json"
python - <<'PY'
import json
from pathlib import Path
r=Path('/workspace/beam-results/native-generator')
assert json.loads((r/'protocol-legacy.json').read_text())=={'protocol':'resident-search-v1'}
assert json.loads((r/'protocol-binding.json').read_text())=={'protocol':'resident-search-v1','loaded_graph_binding':1}
PY
echo CARGO_TEST_OK >> "$root/status.txt"
echo PYTHON_START >> "$root/status.txt"
python rust/beam-python/tests/search_session.py integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so > "$root/pyo3-session.log" 2>&1
python rust/beam-python/tests/interrupt.py integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so > "$root/pyo3-interrupt.log" 2>&1
python -m pytest -q integrations/cayleypy_native/tests --tb=short --basetemp /workspace/tmp/graph-pytest -p no:cacheprovider > "$root/python-tests.log" 2>&1
echo PYTHON_OK >> "$root/status.txt"
echo GPU_PUBLIC_START >> "$root/status.txt"
python tests/public_rust_graph_gpu.py --source "$work" --cutlass /workspace/cutlass --cache "$root/cache" --out "$root/gpu" > "$root/gpu.log" 2>&1
echo GPU_PUBLIC_OK >> "$root/status.txt"
mapfile -t libraries < <(find "$root/cache/builds" -type f -name libbeam_native.so)
if [[ ${#libraries[@]} != 1 ]]; then printf 'expected one library, found %s\n' "${#libraries[@]}"; exit 1; fi
build_dir=$(dirname "${libraries[0]}")
cmake --build "$build_dir" --target native_loaded_graph_hash_tests native_abi_tests --parallel 2 > "$root/cmake-test-build.log" 2>&1
ctest --test-dir "$build_dir" -R '^(native_loaded_graph_hash_tests|native_abi_tests)$' --output-on-failure > "$root/ctest.log" 2>&1
echo GPU_MISMATCH_START >> "$root/status.txt"
python tests/native_loaded_graph_binding_gpu.py --runner "$worker" --library "${libraries[0]}" \
    --extension /workspace/target/release/lib_beam_runtime.so --out "$root/mismatch" > "$root/mismatch.log" 2>&1
echo GPU_MISMATCH_OK >> "$root/status.txt"
verify_source > "$root/source-after.txt"
echo COMPLETE >> "$root/status.txt"
