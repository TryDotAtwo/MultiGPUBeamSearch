#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
mkdir -p "$tmp/build" "$tmp/data" "$tmp/weights"
touch "$tmp/data/puzzle_info.json" "$tmp/data/test.csv" "$tmp/weights/manifest.json"
printf '#!/usr/bin/env bash\nset -euo pipefail\nprintf "rank=%%s world=%%s beam=%%s capacity=%%s exchange=%%s qkv=%%s ff1=%%s ff2=%%s\\n" "$RANK" "$WORLD_SIZE" "$3" "$BEAM_SHARD_CAPACITY_CANDIDATES" "$BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM" "$BEAM_STREAM1_TRANSFORMER_HOPPER_QKV" "$BEAM_STREAM1_TRANSFORMER_HOPPER_FF1" "$BEAM_STREAM1_TRANSFORMER_HOPPER_FF2"\n' > "$tmp/build/production_runner"
printf '#!/usr/bin/env bash\nset -euo pipefail\nprintf "micro=%%s lanes=%%s qkv=%%s\\n" "$BEAM_STREAM1_TRANSFORMER_B_MICRO" "$BEAM_STREAM1_TRANSFORMER_CONCURRENCY" "$BEAM_STREAM1_TRANSFORMER_HOPPER_QKV" > "$BEAM_STREAM_BENCH_REPORT"\n' > "$tmp/build/stream_benchmark"
chmod +x "$tmp/build/production_runner" "$tmp/build/stream_benchmark"
export B300_REPO_DIR="$PWD" B300_RUN_ROOT="$tmp" B300_BUILD_DIR="$tmp/build"
export B300_WEIGHT_DIR="$tmp/weights" B300_DATA_DIR="$tmp/data"
export BEAM_WIDTH=256000000 DEPTH_LIMIT=9

missing=$(mktemp -d)
if B300_DATA_DIR="$missing" B300_WORLD_SIZE=1 bash hpc/b300_fp16.sh s1 > "$tmp/missing.log" 2>&1; then
  echo 'FAIL: missing puzzle inputs accepted' >&2
  exit 1
fi
[[ -z $(find "$tmp" -maxdepth 1 -type d -name 'b300_s1.*' -print -quit) ]]

single_output=$(B300_WORLD_SIZE=1 bash hpc/b300_fp16.sh run)
single_dir=$(printf '%s\n' "$single_output" | sed -n 's/^run_dir=\([^ ]*\).*/\1/p')
[[ -d "$single_dir" ]]
grep -q 'rank=0 world=1 beam=256000000 capacity=8000512 exchange=1000000 qkv=off ff1=off ff2=off' "$single_dir/rank0.log"

dual_output=$(B300_WORLD_SIZE=2 bash hpc/b300_fp16.sh run)
dual_dir=$(printf '%s\n' "$dual_output" | sed -n 's/^run_dir=\([^ ]*\).*/\1/p')
logs=("$dual_dir"/rank*.log)
[[ ${#logs[@]} == 2 ]]
for log in "${logs[@]}"; do
  grep -q 'world=2 beam=256000000 capacity=4000768 exchange=2000000 qkv=off ff1=off ff2=off' "$log"
done

B300_WORLD_SIZE=1 bash hpc/b300_fp16.sh s1
grep -q 'micro=384 lanes=8 qkv=off' "$tmp"/b300_s1.*/gpu0.md
echo 'PASS: B300 profile, rank count, beam alignment, and isolated Stream1 wiring; mock only'
