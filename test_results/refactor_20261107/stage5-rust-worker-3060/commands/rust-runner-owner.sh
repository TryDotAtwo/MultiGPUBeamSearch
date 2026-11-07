set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
installed=/workspace/beam-results/rust-runner-wheel/final/check/checkout/installed
export PYTHONPATH="$installed"
out=/workspace/beam-results/rust-runner-wheel/owner
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
worker="$installed/multigpubeamsearch/_bin/beam-runner"
extension="$installed/multigpubeamsearch/_beam_runtime.abi3.so"
library=/workspace/beam-results/rust-wheel/gpu/build/libbeam_native.so
torch_library=/workspace/beam-results/rust-wheel/gpu/build/libbeam_native_libtorch.so
fixture=/workspace/beam-results/rust-runner-wheel/session/run
sha256sum "$worker" "$extension" "$library" "$torch_library" > "$out/binaries-before.sha256"
cd /workspace/beam-work
python tests/native_search_owner_gpu.py --runner "$worker" --library "$library" --libtorch-library "$torch_library" \
  --extension "$extension" --fixture "$fixture" --out "$out/run" > "$out/owner.log" 2>&1 || { tail -n 120 "$out/owner.log"; exit 1; }
cat "$out/owner.log"
sha256sum "$worker" "$extension" "$library" "$torch_library" > "$out/binaries-after.sha256"
cmp "$out/binaries-before.sha256" "$out/binaries-after.sha256"
echo PACKAGED_RUST_OWNER_GPU_SUCCESS > "$out/success.txt"
