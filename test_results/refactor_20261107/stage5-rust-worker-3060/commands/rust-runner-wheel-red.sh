set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp
out=/workspace/beam-results/rust-runner-wheel/red
mkdir -p "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
set +e
python tests/rust_distribution_check.py --source /workspace/beam-work --out "$out/check" > "$out/test.log" 2>&1
rc=$?
set -e
cat "$out/test.log"
test "$rc" -ne 0
grep -F 'multigpubeamsearch/_bin/beam-runner' "$out/test.log"
printf 'EXPECTED_RED_EXIT=%s\n' "$rc"
