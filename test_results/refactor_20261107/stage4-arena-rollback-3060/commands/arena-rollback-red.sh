set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
out=/workspace/beam-results/arena-rollback/red
mkdir -p "$out"
python - <<'PY'
import hashlib,json,shutil
from pathlib import Path
src=Path('/workspace/beam-work'); dst=Path('/workspace/beam-native-source')
manifest=json.loads(Path('/workspace/current-source-manifest.json').read_text())
changed=[]
for name,digest in manifest.items():
 if name != 'CMakeLists.txt' and not name.startswith(('cuda/','src/','cmake/','tests/','tools/','native/')): continue
 a,b=src/name,dst/name
 assert hashlib.sha256(a.read_bytes()).hexdigest()==digest,name
 if not b.is_file() or hashlib.sha256(b.read_bytes()).hexdigest()!=digest:
  b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b);changed.append(name)
 assert hashlib.sha256(b.read_bytes()).hexdigest()==digest,name
print('NATIVE_SOURCE_SNAPSHOT_VERIFIED',len(changed))
PY
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cmake -S /workspace/beam-native-source -B /workspace/build-work -DBEAM_EXPERIMENTAL_RADIX128=OFF -DBEAM_EXPERIMENTAL_INDEX_SORT=OFF -DBEAM_EXPERIMENTAL_STREAM4_SCAN=OFF -DBEAM_EXPERIMENTAL_OWNER_RADIX=OFF -DBEAM_EXPERIMENTAL_SCORE_BITS=OFF > "$out/configure.log" 2>&1
cmake --build /workspace/build-work --parallel 2 --target static_memory_cuda_tests > "$out/build.log" 2>&1 || { tail -n 80 "$out/build.log"; exit 1; }
cd "$out"
set +e
CUDA_VISIBLE_DEVICES=0 timeout 120s /workspace/build-work/static_memory_cuda_tests > test.log 2>&1
rc=$?
set -e
echo "exit_code=$rc" | tee exit-code.txt
tail -n 15 test.log
if [ "$rc" != 0 ] && grep -q 'union layout failure leaked the arena' test.log; then echo EXPECTED_ROLLBACK_FAILURE > success.txt; else echo RED_TEST_NOT_REPRODUCED; exit 1; fi
