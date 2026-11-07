set -euo pipefail
export PATH=/workspace/.cargo/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/workspace/beam-work/integrations/cayleypy_native
out=/workspace/beam-results/cub-init/session-pairs1
mkdir "$out"
cp /workspace/current-source-manifest.json "$out/source-manifest.json"
cd /workspace/beam-work
sha256sum /workspace/beam-results/cub-init/green1/production_runner /workspace/beam-results/cub-init/green1/libbeam_native.so /workspace/target/release/beam-runner /workspace/target/release/lib_beam_runtime.so /workspace/beam-results/score-bits/default1/production_runner > "$out/binaries-before.sha256"
python tests/native_session_gpu.py --baseline /workspace/beam-results/score-bits/default1/production_runner --rust-runner /workspace/target/release/beam-runner --library /workspace/beam-results/cub-init/green1/libbeam_native.so --extension /workspace/target/release/lib_beam_runtime.so --out "$out/session" > "$out/session.log" 2>&1 || { tail -n 100 "$out/session.log"; exit 1; }
cat "$out/session.log"
for world in 1 2; do
 python tests/native_radix128_pair_gpu.py --world "$world" --control /workspace/beam-results/score-bits/default1/production_runner --radix /workspace/beam-results/cub-init/green1/production_runner --fixture /workspace/beam-results/history/gpu-disk --out "$out/pair-world$world" > "$out/pair-world$world.log" 2>&1 || { tail -n 100 "$out/pair-world$world.log"; exit 1; }
 cat "$out/pair-world$world.log"
done
sha256sum /workspace/beam-results/cub-init/green1/production_runner /workspace/beam-results/cub-init/green1/libbeam_native.so /workspace/target/release/beam-runner /workspace/target/release/lib_beam_runtime.so /workspace/beam-results/score-bits/default1/production_runner > "$out/binaries-after.sha256"
cmp "$out/binaries-before.sha256" "$out/binaries-after.sha256"
echo CUB_INIT_SESSION_PAIRS1_SUCCESS > "$out/success.txt"
