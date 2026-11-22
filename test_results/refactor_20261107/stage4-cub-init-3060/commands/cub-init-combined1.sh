set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
out=/workspace/beam-results/cub-init/combined1
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cmake -S /workspace/beam-native-source -B /workspace/build-work -DBEAM_EXPERIMENTAL_RADIX128=ON -DBEAM_EXPERIMENTAL_INDEX_SORT=ON -DBEAM_EXPERIMENTAL_STREAM4_SCAN=ON -DBEAM_EXPERIMENTAL_OWNER_RADIX=ON -DBEAM_EXPERIMENTAL_SCORE_BITS=ON > "$out/configure.log" 2>&1
cmake --build /workspace/build-work --parallel 2 --target static_memory_cuda_tests score_histogram_cuda_tests union_memory_cuda_tests stream3_cuda_tests stream4_cuda_tests stitched_cuda_tests stream4_index_reducer_cuda_tests radix128_cuda_tests beam_native > "$out/build.log" 2>&1 || { tail -n 100 "$out/build.log"; exit 1; }
sha256sum /workspace/build-work/libbeam_native.so /workspace/target/release/beam-runner /workspace/target/release/lib_beam_runtime.so /workspace/beam-results/score-bits/default1/production_runner > "$out/binaries-before.sha256"
cd "$out"
for gpu in 0 1; do
 for t in static_memory_cuda_tests score_histogram_cuda_tests union_memory_cuda_tests stream4_index_reducer_cuda_tests radix128_cuda_tests; do
  CUDA_VISIBLE_DEVICES=$gpu BEAM_TEST_USE_ARENA_SCRATCH=1 timeout 360s compute-sanitizer --tool initcheck --check-api-memory-access yes --error-exitcode 98 /workspace/build-work/$t > init-$t-gpu$gpu.log 2>&1 || { echo FAILED:init:$t:$gpu; tail -n 35 init-$t-gpu$gpu.log; exit 1; }
  echo PASS:init:$t:$gpu; tail -n 2 init-$t-gpu$gpu.log
 done
 CUDA_VISIBLE_DEVICES=$gpu BEAM_TEST_USE_ARENA_SCRATCH=1 timeout 360s compute-sanitizer --tool memcheck --leak-check full --error-exitcode 98 /workspace/build-work/score_histogram_cuda_tests > memcheck-score-gpu$gpu.log 2>&1 || { tail -n 35 memcheck-score-gpu$gpu.log; exit 1; }
 echo PASS:memcheck:score:$gpu; tail -n 2 memcheck-score-gpu$gpu.log
done
for t in stream3_cuda_tests stream4_cuda_tests stitched_cuda_tests; do
 CUDA_VISIBLE_DEVICES=0,1 /workspace/build-work/$t > $t.log 2>&1 || { tail -n 50 $t.log; exit 1; }
 echo PASS:$t
done
cd /workspace/beam-work
python tests/native_session_gpu.py --baseline /workspace/beam-results/score-bits/default1/production_runner --rust-runner /workspace/target/release/beam-runner --library /workspace/build-work/libbeam_native.so --extension /workspace/target/release/lib_beam_runtime.so --out "$out/session" > "$out/session.log" 2>&1 || { tail -n 100 "$out/session.log"; exit 1; }
cat "$out/session.log"
sha256sum /workspace/build-work/libbeam_native.so /workspace/target/release/beam-runner /workspace/target/release/lib_beam_runtime.so /workspace/beam-results/score-bits/default1/production_runner > "$out/binaries-after.sha256"
cmp "$out/binaries-before.sha256" "$out/binaries-after.sha256"
echo CUB_INIT_COMBINED1_SUCCESS > "$out/success.txt"
