#!/bin/bash
set -Eeuo pipefail
root=/workspace/beam-results/native-generator
base=/workspace/beam-base
work=/workspace/beam-work
mkdir -p "$root" /workspace/tmp "$base" "$work" /workspace/cutlass
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
tar -xzf /workspace/graph-base.tar.gz -C "$base"
tar -xzf /workspace/graph-base.tar.gz -C "$work"
tar -xzf /workspace/graph-cutlass.tar.gz -C /workspace/cutlass
tar -xzf /workspace/graph-overlay.tar.gz -C "$work" --exclude=manifest.json
verify_source() {
python - <<'PY'
import hashlib,json,tarfile
from pathlib import Path
with tarfile.open('/workspace/graph-overlay.tar.gz','r:gz') as tar:
    m=json.load(tar.extractfile('manifest.json'))
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
echo CPU_BUILD_START >> "$root/status.txt"
cat > /workspace/tmp/old-shim.cpp <<'CPP'
#include "native_input.hpp"
beam::input::NativeMoves beam::input::load_p900_moves(const std::filesystem::path& path) {
    NativeMoves result;
    result.data = load_p900_generators(path);
    result.names = load_p900_move_names(path);
    return result;
}
CPP
for shape in 120-128-18 150-160-255 96-112-24; do
    IFS=- read -r logical physical count <<< "$shape"
    defs=(-DBEAM_STATE_LOGICAL_BYTES="$logical" -DBEAM_STATE_PHYSICAL_BYTES="$physical" -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT="$count")
    g++ -std=c++20 -O2 "${defs[@]}" "$work/tests/native_input_tests.cpp" "$work/src/native_input.cpp" -o "/workspace/tmp/input-$shape" >> "$root/cpu-build.log" 2>&1
    g++ -std=c++20 -O2 "${defs[@]}" -DBEAM_BENCH_COMBINED "$work/tests/native_generator_parse_benchmark.cpp" "$work/src/native_input.cpp" -o "/workspace/tmp/bench-new-$shape" >> "$root/cpu-build.log" 2>&1
    g++ -std=c++20 -O2 "${defs[@]}" "$work/tests/native_generator_parse_benchmark.cpp" "$base/src/native_input.cpp" -o "/workspace/tmp/bench-old-$shape" >> "$root/cpu-build.log" 2>&1
    python "$work/tests/native_generator_snapshot_fifo.py" "/workspace/tmp/input-$shape" --state-len "$logical" --moves "$count" --tmpdir /workspace/tmp >> "$root/fifo.log" 2>&1
    python "$work/tests/native_generator_parse_bench.py" --baseline "/workspace/tmp/bench-old-$shape" --candidate "/workspace/tmp/bench-new-$shape" --state-len "$logical" --moves "$count" --out "$root/cpu-$shape" >> "$root/cpu-benchmark.log" 2>&1
done
defs=(-DBEAM_STATE_LOGICAL_BYTES=120 -DBEAM_STATE_PHYSICAL_BYTES=128 -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=18)
g++ -std=c++20 -O2 "${defs[@]}" -I "$work/src" "$work/tests/native_input_tests.cpp" "$base/src/native_input.cpp" /workspace/tmp/old-shim.cpp -o /workspace/tmp/input-old-shim >> "$root/cpu-build.log" 2>&1
if python "$work/tests/native_generator_snapshot_fifo.py" /workspace/tmp/input-old-shim --state-len 120 --moves 18 --tmpdir /workspace/tmp > "$root/fifo-old-red.log" 2>&1; then
    echo 'old two-read shim unexpectedly passed FIFO' >> "$root/fifo-old-red.log"; exit 1
fi
grep -q 'TimeoutExpired' "$root/fifo-old-red.log"
mkdir -p "$root/input-tests"
(cd "$root/input-tests" && /workspace/tmp/input-120-128-18) > "$root/native-input-tests.log" 2>&1
echo CPU_TEST_OK >> "$root/status.txt"
cd "$work"
echo CARGO_TEST_START >> "$root/status.txt"
cargo test --workspace --locked > "$root/cargo-tests.log" 2>&1
cargo build --release --locked -p beam-python -p beam-cli > "$root/cargo-build.log" 2>&1
cp /workspace/target/release/lib_beam_runtime.so integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so
mkdir -p integrations/cayleypy_native/multigpubeamsearch/_bin
cp /workspace/target/release/beam-runner integrations/cayleypy_native/multigpubeamsearch/_bin/beam-runner
sha256sum integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.abi3.so integrations/cayleypy_native/multigpubeamsearch/_bin/beam-runner > "$root/rust-binaries.sha256"
echo CARGO_TEST_OK >> "$root/status.txt"
echo PYTHON_START >> "$root/status.txt"
python -m pytest -q integrations/cayleypy_native/tests --tb=short --basetemp /workspace/tmp/generator-pytest -p no:cacheprovider > "$root/python-tests.log" 2>&1
echo PYTHON_OK >> "$root/status.txt"
echo GPU_PUBLIC_START >> "$root/status.txt"
python tests/public_rust_graph_gpu.py --source "$work" --cutlass /workspace/cutlass --cache "$root/cache" --out "$root/gpu" > "$root/gpu.log" 2>&1
echo GPU_PUBLIC_OK >> "$root/status.txt"
echo GPU_PAIR_BUILD_START >> "$root/status.txt"
python - <<'PY' > "$root/nccl.json"
import json
from multigpubeamsearch.build import discover_nccl
print(json.dumps(discover_nccl()))
PY
readarray -t nccl < <(python -c 'import json; x=json.load(open("/workspace/beam-results/native-generator/nccl.json")); print(x["include_dir"]); print(x["library"])')
nccl_dir=$(dirname "${nccl[1]}")
for variant in base candidate; do
    source="$base"; if [[ "$variant" == candidate ]]; then source="$work"; fi
    cmake -S "$source" -B "/workspace/build-$variant" -DCMAKE_BUILD_TYPE=Release -DCUTLASS_DIR=/workspace/cutlass \
        -DBEAM_STATE_LOGICAL_BYTES=96 -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=24 -DBEAM_CUDA_ARCHITECTURES=86 \
        -DBEAM_ENABLE_DEBUG=ON -DBEAM_ENABLE_DEBUG_LOGS=ON -DBEAM_ENABLE_DEPTH_LOGS=OFF \
        -DNCCL_INCLUDE_DIR="${nccl[0]}" -DNCCL_LIBRARY="${nccl[1]}" -DCMAKE_BUILD_RPATH="$nccl_dir" > "$root/configure-$variant.log" 2>&1
    cmake --build "/workspace/build-$variant" --target production_runner --parallel 2 > "$root/build-$variant.log" 2>&1
    cp "/workspace/build-$variant/production_runner" "$root/production_runner-$variant"
done
python - <<'PY'
import json
from pathlib import Path
from public_model_snapshot_gpu import graph_fixture
from multigpubeamsearch.hamming import write_hamming_artifact
root=Path('/workspace/beam-results/native-generator/fixture')
root.mkdir()
graph,contract,moves=graph_fixture()
(root/'puzzle.json').write_text(json.dumps(contract.to_puzzle_info())+'\n')
write_hamming_artifact(contract,root)
PY
echo GPU_PAIR_START >> "$root/status.txt"
python tests/native_radix128_pair_gpu.py --control "$root/production_runner-base" --radix "$root/production_runner-candidate" --fixture "$root/fixture" --out "$root/pair" > "$root/pair.log" 2>&1
echo GPU_PAIR_OK >> "$root/status.txt"
verify_source > "$root/source-after.txt"
echo COMPLETE >> "$root/status.txt"
