set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
out=/workspace/beam-results/cub-init/red1
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
cmake --build /workspace/build-work --parallel 2 --target static_memory_cuda_tests score_histogram_cuda_tests > "$out/build.log" 2>&1 || { tail -n 80 "$out/build.log"; exit 1; }
cd "$out"
set +e
CUDA_VISIBLE_DEVICES=0 timeout 360s compute-sanitizer --tool initcheck --check-api-memory-access yes --error-exitcode 98 /workspace/build-work/static_memory_cuda_tests > static-init.log 2>&1
static_rc=$?
CUDA_VISIBLE_DEVICES=0 BEAM_TEST_USE_ARENA_SCRATCH=1 timeout 360s compute-sanitizer --tool initcheck --error-exitcode 98 /workspace/build-work/score_histogram_cuda_tests > arena-score-init.log 2>&1
arena_rc=$?
set -e
printf 'static_rc=%s\narena_rc=%s\n' "$static_rc" "$arena_rc" | tee exit-codes.txt
tail -n 8 static-init.log
tail -n 8 arena-score-init.log
if [ "$static_rc" = 98 ] && [ "$arena_rc" = 98 ] && grep -qi uninitialized static-init.log && grep -qi uninitialized arena-score-init.log; then echo EXPECTED_REPRODUCTION > success.txt; else echo RED_TEST_NOT_REPRODUCED; exit 1; fi
