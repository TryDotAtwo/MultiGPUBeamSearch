set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
out=/workspace/beam-results/arena-rollback/green
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
cmake --build /workspace/build-work --parallel 2 --target static_memory_cuda_tests score_histogram_cuda_tests union_memory_cuda_tests > "$out/build.log" 2>&1 || { tail -n 100 "$out/build.log"; exit 1; }
cd "$out"
for gpu in 0 1; do
 CUDA_VISIBLE_DEVICES=$gpu timeout 360s compute-sanitizer --tool memcheck --leak-check full --error-exitcode 98 /workspace/build-work/static_memory_cuda_tests > memcheck-static-gpu$gpu.log 2>&1 || { tail -n 50 memcheck-static-gpu$gpu.log; exit 1; }
 echo PASS:memcheck:static:$gpu; tail -n 3 memcheck-static-gpu$gpu.log
 CUDA_VISIBLE_DEVICES=$gpu timeout 360s compute-sanitizer --tool initcheck --check-api-memory-access yes --error-exitcode 98 /workspace/build-work/static_memory_cuda_tests > init-static-gpu$gpu.log 2>&1 || { tail -n 50 init-static-gpu$gpu.log; exit 1; }
 echo PASS:init:static:$gpu; tail -n 2 init-static-gpu$gpu.log
 for t in score_histogram_cuda_tests union_memory_cuda_tests; do
  CUDA_VISIBLE_DEVICES=$gpu BEAM_TEST_USE_ARENA_SCRATCH=1 timeout 360s compute-sanitizer --tool memcheck --leak-check full --error-exitcode 98 /workspace/build-work/$t > memcheck-$t-gpu$gpu.log 2>&1 || { tail -n 50 memcheck-$t-gpu$gpu.log; exit 1; }
  echo PASS:memcheck:$t:$gpu; tail -n 2 memcheck-$t-gpu$gpu.log
 done
done
echo ARENA_ROLLBACK_GREEN_SUCCESS > success.txt
