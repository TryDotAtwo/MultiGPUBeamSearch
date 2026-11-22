set -euo pipefail
out=/workspace/beam-results/native-abi/red
mkdir -p "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
runner=/workspace/beam-results/rust-runner-wheel/final/check/checkout/installed/multigpubeamsearch/_bin/beam-runner
library=/workspace/beam-results/rust-wheel/gpu/build/libbeam_native.so
cd /workspace/beam-work
set +e
python tests/native_abi_probe.py --runner "$runner" --library "$library" --state-len 96 --storage-len 112 --move-count 24 --out "$out/result.json" > "$out/test.log" 2>&1
rc=$?
set -e
cat "$out/test.log"
test "$rc" -ne 0
printf 'EXPECTED_RED_EXIT=%s\n' "$rc"
