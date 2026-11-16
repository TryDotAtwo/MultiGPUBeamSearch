#!/bin/bash
set -Eeuo pipefail
work=/workspace/rust-cluster-src
out=/workspace/rust-cluster-results
mkdir -p "$work" "$out" /workspace/tmp
finish() {
  code=$?
  trap - EXIT
  printf 'EXIT_CODE=%s\n' "$code" >> "$out/status.txt"
  cp /workspace/source-manifest.json /workspace/run.sh "$out/"
  tar -czf /workspace/rust-cluster-evidence.tar.gz -C /workspace rust-cluster-results
  sha256sum /workspace/rust-cluster-evidence.tar.gz > /workspace/rust-cluster-evidence.sha256
  exit "$code"
}
trap finish EXIT
tar -xzf /workspace/base.tar.gz -C "$work"
cp -a /workspace/overlay/. "$work/"
verify_source() {
  python - "$work" /workspace/source-manifest.json <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); m=json.loads(Path(sys.argv[2]).read_text())
expected=dict(m['baseline']); expected.update(m['overlays'])
for name,digest in sorted(expected.items()):
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert actual==digest,(name,actual,digest)
print('verified source files',len(expected),'base',m['base_commit'])
PY
}
verify_source > "$out/source-before.txt"
export TMPDIR=/workspace/tmp CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target
export PATH="$CARGO_HOME/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$work/integrations/cayleypy_native:$work/tests" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export REQUIRE_RUST_CLUSTER=1 CARGO_BUILD_JOBS=4
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$out/gpus.csv"
python -c 'import platform,torch;print(platform.platform());print(torch.__version__);print(torch.version.cuda);print("GPU_COUNT",torch.cuda.device_count());assert torch.cuda.device_count()>=2' > "$out/software.txt"
echo BOOTSTRAP >> "$out/status.txt"
python -m pip install -q pytest 'cayleypy>=0.1.0' 'cmake>=3.28' ninja > "$out/pip.log" 2>&1
if ! command -v cargo >/dev/null; then
  curl --retry 4 --retry-all-errors --retry-delay 2 --connect-timeout 15 --max-time 120 --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /workspace/tmp/rustup.sh
  sh /workspace/tmp/rustup.sh -y --profile minimal --no-modify-path > "$out/rustup.log" 2>&1
fi
cd "$work"
echo RUST_TESTS >> "$out/status.txt"
timeout 600 cargo test --workspace --locked > "$out/rust-tests.log" 2>&1
echo RUST_EXTENSION >> "$out/status.txt"
timeout 300 cargo build --locked -p beam-python > "$out/rust-build.log" 2>&1
cp "$CARGO_TARGET_DIR/debug/lib_beam_runtime.so" "$work/integrations/cayleypy_native/multigpubeamsearch/_beam_runtime.so"
echo CLUSTER_TESTS >> "$out/status.txt"
timeout 300 python -m pytest -q integrations/cayleypy_native/tests/test_cluster.py integrations/cayleypy_native/tests/test_cluster_worker.py integrations/cayleypy_native/tests/test_cluster_rust.py --tb=short --basetemp /workspace/tmp/cluster -p no:cacheprovider > "$out/cluster-tests.log" 2>&1
echo PYTHON_SUITE >> "$out/status.txt"
timeout 500 python -m pytest -q integrations/cayleypy_native/tests --tb=short --basetemp /workspace/tmp/suite -p no:cacheprovider > "$out/python-tests.log" 2>&1
for engine in python rust; do
  echo "NCCL_${engine}" >> "$out/status.txt"
  timeout 150 torchrun --standalone --nproc-per-node=2 integrations/cayleypy_native/tests/cluster_rank_nccl.py --engine "$engine" --shared-dir "$out/nccl-$engine" > "$out/nccl-$engine.log" 2>&1
  python - "$out/nccl-$engine" <<'PY' >> "$out/nccl-verified.txt"
import json,sys
from pathlib import Path
files=sorted(Path(sys.argv[1]).glob('*/attempt-0/rank-*/nccl-result.json'))
assert len(files)==2,files
rows=[json.loads(p.read_text()) for p in files]
assert sorted(row['rank'] for row in rows)==[0,1]
assert all(row['world']==2 and row['sum']==3 for row in rows)
assert len({row['id_file'] for row in rows})==1
print(sys.argv[1],json.dumps(rows))
PY
done
verify_source > "$out/source-after.txt"
cmp "$out/source-before.txt" "$out/source-after.txt"
echo COMPLETE >> "$out/status.txt"
