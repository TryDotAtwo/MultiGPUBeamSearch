set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
installed=/workspace/beam-results/rust-runner-wheel/final/check/checkout/installed
export PYTHONPATH="$installed"
out=/workspace/beam-results/rust-runner-wheel/session
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
worker="$installed/multigpubeamsearch/_bin/beam-runner"
extension="$installed/multigpubeamsearch/_beam_runtime.abi3.so"
library=/workspace/beam-results/rust-wheel/gpu/build/libbeam_native.so
baseline=/workspace/beam-results/cub-init/green1/production_runner
sha256sum "$worker" "$extension" "$library" "$baseline" > "$out/binaries-before.sha256"
cd /workspace/beam-work
python tests/native_session_gpu.py --baseline "$baseline" --rust-runner "$worker" --library "$library" \
  --extension "$extension" --out "$out/run" > "$out/session.log" 2>&1 || { tail -n 100 "$out/session.log"; exit 1; }
cat "$out/session.log"
sha256sum "$worker" "$extension" "$library" "$baseline" > "$out/binaries-after.sha256"
cmp "$out/binaries-before.sha256" "$out/binaries-after.sha256"
echo PACKAGED_RUST_RUNNER_GPU_SUCCESS > "$out/success.txt"
