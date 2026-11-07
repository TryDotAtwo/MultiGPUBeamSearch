set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
out=/workspace/beam-results/rust-runner-wheel/libtorch-build
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
python - <<'PY' > "$out/source-match.json"
import hashlib,json
from pathlib import Path
old=Path('/workspace/beam-results/rust-wheel/final/check/checkout/installed/multigpubeamsearch/_native_source')
root=Path('/workspace/beam-work')
manifest=json.loads((old/'source_manifest.json').read_text())
prefix=('src/','cuda/','tools/','cmake/','native/','rust/')
selected=[name for name in manifest if name in ('CMakeLists.txt','Cargo.toml','Cargo.lock') or name.startswith(prefix)]
for name in selected:
    expected=hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    assert manifest[name]==expected,name
print(json.dumps(dict(matching_compilation_source_files=len(selected),scope='all source manifest entries under compile roots'),indent=2))
PY
cmake --build /workspace/beam-results/rust-wheel/gpu/build -j 2 --target beam_native_libtorch > "$out/build.log" 2>&1 || { tail -n 100 "$out/build.log"; exit 1; }
sha256sum /workspace/beam-results/rust-wheel/gpu/build/libbeam_native_libtorch.so > "$out/library.sha256"
cat "$out/source-match.json"
cat "$out/library.sha256"
echo LIBTORCH_FROM_BUNDLED_SOURCE_SUCCESS > "$out/success.txt"
