set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native TORCH_CUDA_ARCH_LIST=8.6
out=/workspace/beam-results/native-torch-identity/real
cd /workspace/beam-work
python /workspace/native-torch-identity-real.py > /workspace/beam-results/native-torch-identity-driver.log 2>&1 || { cat /workspace/beam-results/native-torch-identity-driver.log; exit 1; }
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
extension=/workspace/beam-results/native-library/distribution4/check/checkout/installed/multigpubeamsearch/_beam_runtime.abi3.so
for kind in cutlass libtorch; do
  readarray -t paths < <(python - "$out/result.json" "$kind" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))[sys.argv[2]]
for key in ('runner','library','worker'):print(data[key])
PY
)
  executor=native_cuda_graph
  if [ "$kind" = libtorch ]; then executor=libtorch_eager; fi
  sha256sum "${paths[@]}" "$extension" > "$out/$kind/binaries-before.sha256"
  python tests/native_session_gpu.py --baseline "${paths[0]}" --rust-runner "${paths[2]}" --library "${paths[1]}" \
    --extension "$extension" --executor "$executor" --out "$out/$kind/session" > "$out/$kind/session.log" 2>&1 || { tail -n 100 "$out/$kind/session.log"; exit 1; }
  cat "$out/$kind/session.log"
  sha256sum "${paths[@]}" "$extension" > "$out/$kind/binaries-after.sha256"
  cmp "$out/$kind/binaries-before.sha256" "$out/$kind/binaries-after.sha256"
done
echo NATIVE_TORCH_IDENTITY_REAL_SUCCESS > "$out/success.txt"
