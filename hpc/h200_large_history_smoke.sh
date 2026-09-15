#!/usr/bin/env bash
# Regression: two 370M-parent history slots must allocate and receive real D2H data.
set -euo pipefail
cd /workspace/MGBFS
log=$(mktemp /workspace/large_history_smoke.XXXXXX)
BEAM_HISTORY_CHUNKED_PIN=1 BEAM_WIDTH=740000000 DEPTH_LIMIT=6 \
  timeout 180 bash hpc/h200_dual_large_beam.sh > "$log" 2>&1
dir=$(sed -n 's/^run_dir=//p' "$log")
for rank in 0 1; do
  grep -q 'depth_done=0 .*next_frontier_size=12 ' "$dir/rank${rank}.log"
  # 469 unique states split as235/234 (verified against the small-beam control).
  expected=235; if ((rank == 1)); then expected=234; fi
  grep -q "depth_done=1 .*next_frontier_size=$expected " "$dir/rank${rank}.log"
  # Assert the actual boundary crossing, not an invariant survivor count across profiles.
  awk '/depth_done=5 / {for(i=1;i<=NF;i++) if($i ~ /^next_frontier_size=/) {
    split($i,a,"="); if(a[2]>33554432) crossed=1
  }} END {exit !crossed}' "$dir/rank${rank}.log"
done
echo "PASS large history allocation and two-rank D2H: $dir"
