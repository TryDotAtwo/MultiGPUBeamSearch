set -eu
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_TARGET_DIR=/workspace/target
out=/workspace/beam-results/rust-wheel/red
mkdir -p "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
set +e
python tests/rust_distribution_check.py --source /workspace/beam-work --out "$out/check" >"$out/test.log" 2>&1
rc=$?
set -e
cat "$out/test.log"
cat "$out/check/build.log"
printf 'RED_EXIT_CODE=%s\n' "$rc"
test "$rc" -ne 0
