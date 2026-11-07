set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp
export TORCH_CUDA_ARCH_LIST=8.6
out=/workspace/beam-results/rust-wheel/gpu
installed=/workspace/beam-results/rust-wheel/final/check/checkout/installed
export PYTHONPATH="$installed"
source="$installed/multigpubeamsearch/_native_source"
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
readarray -t NCCL < <(python -c 'from multigpubeamsearch.build import discover_nccl; n=discover_nccl();print(n["include_dir"]);print(n["library"])')
TORCH_PREFIX=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
cmake -S "$source" -B "$out/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_DIR=/workspace/cutlass -DBEAM_CUDA_ARCHITECTURES=86 \
  -DBEAM_STATE_LOGICAL_BYTES=96 -DBEAM_MOVE_COUNT=24 \
  -DNCCL_INCLUDE_DIR="${NCCL[0]}" -DNCCL_LIBRARY="${NCCL[1]}" \
  -DBEAM_ENABLE_LIBTORCH_STREAM1=ON -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
  > "$out/configure.log" 2>&1 || { tail -n 80 "$out/configure.log"; exit 1; }
cmake --build "$out/build" -j 2 --target production_runner beam_native > "$out/build.log" 2>&1 || { tail -n 100 "$out/build.log"; exit 1; }
extension="$installed/multigpubeamsearch/_beam_runtime.abi3.so"
sha256sum "$out/build/production_runner" "$out/build/libbeam_native.so" "$extension" \
  /workspace/target/release/beam-runner /workspace/beam-results/cub-init/green1/production_runner > "$out/binaries-before.sha256"
cd /workspace/beam-work
python tests/native_session_gpu.py \
  --baseline /workspace/beam-results/cub-init/green1/production_runner \
  --rust-runner /workspace/target/release/beam-runner --library "$out/build/libbeam_native.so" \
  --extension "$extension" --out "$out/run" > "$out/session.log" 2>&1 || { tail -n 100 "$out/session.log"; exit 1; }
cat "$out/session.log"
sha256sum "$out/build/production_runner" "$out/build/libbeam_native.so" "$extension" \
  /workspace/target/release/beam-runner /workspace/beam-results/cub-init/green1/production_runner > "$out/binaries-after.sha256"
cmp "$out/binaries-before.sha256" "$out/binaries-after.sha256"
echo RUST_WHEEL_GPU_SUCCESS > "$out/success.txt"
