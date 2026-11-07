set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp
installed=/workspace/beam-results/native-library/distribution3/check/checkout/installed
export PYTHONPATH="$installed" TORCH_CUDA_ARCH_LIST=8.6
out=/workspace/beam-results/native-library/actual
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
python tests/native_library_gpu.py --source "$installed/multigpubeamsearch/_native_source" --cutlass /workspace/cutlass \
  --out "$out/build" > "$out/build.log" 2>&1 || { tail -n 150 "$out/build.log"; find "$out/build/logs" -name '*.log' -exec tail -n 40 {} \; 2>/dev/null; exit 1; }
cat "$out/build.log"
readarray -t paths < <(python - "$out/build/result.json" <<'PY'
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
echo NATIVE_LIBRARY_ACTUAL_GPU_SUCCESS > "$out/success.txt"
