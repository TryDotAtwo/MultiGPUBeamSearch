set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
out=/workspace/beam-results/arena-rollback/session
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cmake --build /workspace/build-work --parallel 2 --target beam_native > "$out/build.log" 2>&1 || { tail -n 80 "$out/build.log"; exit 1; }
sha256sum /workspace/build-work/libbeam_native.so /workspace/target/release/beam-runner /workspace/target/release/lib_beam_runtime.so /workspace/beam-results/cub-init/green1/production_runner > "$out/binaries-before.sha256"
cd /workspace/beam-work
python tests/native_session_gpu.py --baseline /workspace/beam-results/cub-init/green1/production_runner --rust-runner /workspace/target/release/beam-runner --library /workspace/build-work/libbeam_native.so --extension /workspace/target/release/lib_beam_runtime.so --out "$out/run" > "$out/session.log" 2>&1 || { tail -n 100 "$out/session.log"; exit 1; }
cat "$out/session.log"
sha256sum /workspace/build-work/libbeam_native.so /workspace/target/release/beam-runner /workspace/target/release/lib_beam_runtime.so /workspace/beam-results/cub-init/green1/production_runner > "$out/binaries-after.sha256"
cmp "$out/binaries-before.sha256" "$out/binaries-after.sha256"
echo ARENA_ROLLBACK_SESSION_SUCCESS > "$out/success.txt"
