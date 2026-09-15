#!/usr/bin/env bash
# Regression: two 370M-parent history slots must allocate and receive real D2H data.
set -euo pipefail
cd /workspace/MGBFS
log=$(mktemp /workspace/large_history_smoke.XXXXXX)
BEAM_HISTORY_CHUNKED_PIN=1 BEAM_WIDTH=740000000 DEPTH_LIMIT=2 \
  timeout 180 bash hpc/h200_dual_large_beam.sh > "$log" 2>&1
dir=$(sed -n 's/^run_dir=//p' "$log")
for rank in 0 1; do
  grep -q 'depth_done=0 .*next_frontier_size=12 ' "$dir/rank${rank}.log"
  grep -q 'depth_done=1 .*next_frontier_size=235 ' "$dir/rank${rank}.log"
done
echo "PASS large history allocation and two-rank D2H: $dir"
