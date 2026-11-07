set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp CARGO_NET_OFFLINE=true
out=/workspace/beam-results/rust-wheel/tamper
mkdir -p "$out/extracted" "$out/wheels"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
tar -xzf /workspace/beam-results/rust-wheel/final/check/sdist/multigpubeamsearch-0.1.0.dev1.tar.gz -C "$out/extracted"
project="$out/extracted/multigpubeamsearch-0.1.0.dev1"
printf '\n// Test-only source tampering.\n' >> "$project/multigpubeamsearch/_native_source/cuda/cnot_domain.cu"
set +e
python -m pip wheel --no-deps --no-index --no-build-isolation --wheel-dir "$out/wheels" "$project" > "$out/pip.log" 2>&1
rc=$?
set -e
test "$rc" -ne 0
grep -F 'source distribution contains modified native source: cuda/cnot_domain.cu' "$out/pip.log"
printf 'EXPECTED_REJECTION_EXIT=%s\n' "$rc" > "$out/result.txt"
cat "$out/result.txt"
