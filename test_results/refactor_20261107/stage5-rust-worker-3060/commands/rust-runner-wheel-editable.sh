set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup CARGO_TARGET_DIR=/workspace/target TMPDIR=/workspace/tmp CARGO_NET_OFFLINE=true
unset PYTHONPATH
out=/workspace/beam-results/rust-runner-wheel/editable
mkdir -p "$out/repository"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
tar --exclude='__pycache__' --exclude='.pytest_cache' --exclude='target' -cf - \
  Cargo.toml Cargo.lock CMakeLists.txt rust integrations/cayleypy_native tools cuda src cmake third_party profiles native tests | tar -xf - -C "$out/repository"
python -m venv --system-site-packages "$out/venv"
"$out/venv/bin/python" -m pip install --no-deps --no-index --no-build-isolation \
  -e "$out/repository/integrations/cayleypy_native" > "$out/install.log" 2>&1 || { tail -n 100 "$out/install.log"; exit 1; }
cd "$out"
"$out/venv/bin/python" - <<'PY' > "$out/import.log" 2>&1
import json,subprocess
import multigpubeamsearch as p,cayleypy_native as c
import multigpubeamsearch._beam_runtime as r
from multigpubeamsearch.rust_control import worker_executable
assert p.NativeOptions is c.NativeOptions and r.ABI_VERSION==1
worker=worker_executable()
assert json.loads(subprocess.check_output([str(worker),'--resident-protocol-info']))=={'protocol':'resident-search-v1'}
print('EDITABLE_RUST_WORKER_OK',p.__file__,r.__file__,worker)
PY
cat "$out/import.log"
echo RUST_RUNNER_EDITABLE_SUCCESS > "$out/success.txt"
