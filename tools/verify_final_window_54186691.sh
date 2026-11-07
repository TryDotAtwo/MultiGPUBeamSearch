#!/usr/bin/env bash
# Diagnostic only. Run sequentially after the completed stats cohort.
set -eu
cd /workspace/network_regression_54186691
export PYTHONPATH=.
export CUTLASS_DIR=/workspace/cutlass
python=/venv/main/bin/python
nccl=/workspace/nccl225_54186691/nvidia/nccl/lib
"$python" evidence/check_canonical_native_status_54186691.py --tag nccl225stats54186691 \
  > /workspace/native_status_GREEN_54186691.log 2>&1
tail -6 /workspace/native_status_GREEN_54186691.log
"$python" evidence/run_vast_asymmetric_numeric_20261002.py --root "$PWD" --build-dir build \
  --run-tag canonical54186691 --nccl-dir "$nccl" > /workspace/asymmetric_numeric_54186691.log 2>&1
tail -8 /workspace/asymmetric_numeric_54186691.log
"$python" evidence/run_vast_dual_neural_20261002.py --root "$PWD" --build-dir build \
  --run-tag unsolved54186691 --nccl-dir "$nccl" --puzzle-id 1000 --max-depth 8 \
  --timeout 180 --final-split 0 > /workspace/unsolved1000_54186691.log 2>&1
tail -15 /workspace/unsolved1000_54186691.log
"$python" - <<'PY' > /workspace/unsolved_native_status_54186691.log
import json
from pathlib import Path
e = Path('evidence')
failed = e / 'dual_asymmetric_numeric_depth3_canonical54186691'
assert not list(failed.glob('rank-status/rank-*.json')), 'failure published successful native status'
run = e / 'dual_neural_p1000_unsolved54186691'
for rank in (0, 1):
    status = json.loads((run / f'rank-status/rank-{rank}.json').read_text())
    assert status == dict(rank=rank, puzzle_id=1000, exit_code=0, status='unsolved', completed_depths=8)
    rows = [json.loads(line) for line in (run / f'predict_stats.rank{rank}.jsonl').read_text().splitlines()]
    assert [row['depth'] for row in rows] == list(range(8))
    assert all(row['rank'] == row['device_local_rank'] == rank for row in rows)
print('PASS two unsolved terminal statuses/depth traces; no successful status on numerical failure')
PY
cat /workspace/unsolved_native_status_54186691.log
timeout 300 "$python" -m pytest -q tests > /workspace/full_canonical_final_54186691.log 2>&1
tail -6 /workspace/full_canonical_final_54186691.log
sha256sum build/production_runner build/stream1_gemm_iterator_probe
