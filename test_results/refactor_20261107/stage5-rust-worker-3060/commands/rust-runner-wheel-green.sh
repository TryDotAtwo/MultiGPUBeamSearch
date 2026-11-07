set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp
out=/workspace/beam-results/rust-runner-wheel/final
mkdir -p "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
set +e
python tests/rust_distribution_check.py --source /workspace/beam-work --out "$out/check" > "$out/test.log" 2>&1
rc=$?
set -e
cat "$out/test.log"
if [ "$rc" -ne 0 ]; then
  find "$out/check" -name '*.log' -type f -exec sh -c 'echo "--- $1"; tail -n 80 "$1"' _ {} \;
fi
exit "$rc"
