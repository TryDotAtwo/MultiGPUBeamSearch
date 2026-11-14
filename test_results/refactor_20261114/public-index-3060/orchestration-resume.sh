#!/usr/bin/env bash
set -euo pipefail
root=/workspace/beam-results/public-index-20261114
trap 'rc=$?; printf "RETRY_FAILED rc=%s line=%s\n" "$rc" "$LINENO" >> "$root/status.txt"' ERR
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/rust-target
export TMPDIR=/workspace/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
export PATH="$CARGO_HOME/bin:/usr/local/cuda/bin:$PATH" CARGO_BUILD_JOBS=2 TORCH_CUDA_ARCH_LIST=8.6
cd /workspace/beam-work
test "$(git rev-parse HEAD)" = 84f8d308ea26dd6842c08f4973fe155d1919628e
test -f /workspace/public-index-legacy.py
test -f /workspace/prior-rust.whl
test ! -e "$root/legacy-compat-attempt1.log"
cp "$root/legacy-compat.log" "$root/legacy-compat-attempt1.log"
echo RETRY1_START >> "$root/status.txt"
echo LEGACY_COMPAT_START >> "$root/status.txt"
python /workspace/public-index-legacy.py > "$root/legacy-compat.log" 2>&1
python -c 'import json; from multigpubeamsearch._beam_runtime import validate_options; assert json.loads(validate_options("{\"experimental_index_sort\":true}"))["experimental_index_sort"] is True; print("CURRENT_RUST_EXTENSION_ON_ACCEPTED")' >> "$root/legacy-compat.log" 2>&1
echo LEGACY_COMPAT_OK >> "$root/status.txt"
sha256sum CMakeLists.txt Cargo.toml Cargo.lock rust/beam-core/src/config.rs \
 integrations/cayleypy_native/multigpubeamsearch/{options.py,build.py,native_library.py,rust_control.py,_beam_runtime.abi3.so,_bin/beam-runner} \
 tests/{public_rust_session_gpu.py,public_index_sort_pair_gpu.py} > "$root/inputs-before.sha256"
echo CPU_TESTS_START >> "$root/status.txt"
python -m pytest -q integrations/cayleypy_native/tests --basetemp /workspace/tmp/index-pytest -p no:cacheprovider > "$root/cpu-tests.log" 2>&1
echo CPU_TESTS_OK >> "$root/status.txt"
for backend in cutlass libtorch; do
  echo "GPU_${backend}_START" >> "$root/status.txt"
  python tests/public_rust_session_gpu.py --source /workspace/beam-work --cutlass /workspace/cutlass \
    --cache "$root/cache-$backend" --out "$root/gpu-$backend" --backend "$backend" \
    --experimental-parallel-nccl-identity --experimental-index-sort --exercise-index-change > "$root/gpu-$backend.log" 2>&1
  echo "GPU_${backend}_OK" >> "$root/status.txt"
done
echo PAIR_START >> "$root/status.txt"
python tests/public_index_sort_pair_gpu.py --source /workspace/beam-work --cutlass /workspace/cutlass \
  --cache "$root/cache-cutlass" --out "$root/paired" > "$root/paired.log" 2>&1
echo PAIR_OK >> "$root/status.txt"
sha256sum CMakeLists.txt Cargo.toml Cargo.lock rust/beam-core/src/config.rs \
 integrations/cayleypy_native/multigpubeamsearch/{options.py,build.py,native_library.py,rust_control.py,_beam_runtime.abi3.so,_bin/beam-runner} \
 tests/{public_rust_session_gpu.py,public_index_sort_pair_gpu.py} > "$root/inputs-after.sha256"
cmp "$root/inputs-before.sha256" "$root/inputs-after.sha256"
echo RUN_COMPLETE >> "$root/status.txt"
