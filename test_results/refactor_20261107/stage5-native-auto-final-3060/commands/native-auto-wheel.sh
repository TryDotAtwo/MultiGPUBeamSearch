set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp
installed=/workspace/beam-results/native-auto/distribution/check/checkout/installed
export PYTHONPATH="$installed" TORCH_CUDA_ARCH_LIST=8.6
out=/workspace/beam-results/native-auto/wheel
cd /workspace/beam-work
python /workspace/native-auto-wheel.py > /workspace/beam-results/native-auto-wheel-driver.log 2>&1 || { cat /workspace/beam-results/native-auto-wheel-driver.log; exit 1; }
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
readarray -t paths < <(python - "$out/result.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
for key in ('runner','library','worker'):print(data[key])
PY
)
extension="$installed/multigpubeamsearch/_beam_runtime.abi3.so"
sha256sum "${paths[@]}" "$extension" > "$out/binaries-before.sha256"
python tests/native_session_gpu.py --baseline "${paths[0]}" --rust-runner "${paths[2]}" --library "${paths[1]}" \
  --extension "$extension" --out "$out/session" > "$out/session.log" 2>&1 || { tail -n 100 "$out/session.log"; exit 1; }
cat "$out/session.log"
sha256sum "${paths[@]}" "$extension" > "$out/binaries-after.sha256"
cmp "$out/binaries-before.sha256" "$out/binaries-after.sha256"
echo NATIVE_AUTO_INSTALLED_WHEEL_SUCCESS > "$out/success.txt"
