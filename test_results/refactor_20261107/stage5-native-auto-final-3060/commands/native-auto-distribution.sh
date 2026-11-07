set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp
out=/workspace/beam-results/native-auto/distribution
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
python tests/rust_distribution_check.py --source /workspace/beam-work --out "$out/check" > "$out/test.log" 2>&1 || { cat "$out/test.log"; find "$out/check" -name '*.log' -type f -exec sh -c 'tail -n 80 "$1"' _ {} \;; exit 1; }
cat "$out/test.log"
echo NATIVE_AUTO_DISTRIBUTION_SUCCESS > "$out/success.txt"
