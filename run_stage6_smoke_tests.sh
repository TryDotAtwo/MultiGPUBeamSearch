#!/bin/bash
set -e

echo "=== Stage6 Dispatcher Skeleton Smoke Tests ==="
cd /workspace/CayleyBeam100H100

echo "Step 1: Build CUDA extension..."
python setup.py build_ext --inplace

echo "Step 2: Install pytest..."
python -m pip install pytest -q

echo "Step 3: Run static architecture tests..."
python -m pytest tests/test_architecture_v6_static.py -q

echo "Step 4: Stream2 reference smoke..."
python tests/stream2_reference_smoke.py

echo "Step 5: Final materialization smoke..."
python tests/final_materialization_smoke.py

echo "Step 6: Stream3 dedup smoke..."
python tests/stream3_dedup_smoke.py

echo "Step 7: Stream4 shard smoke..."
python tests/stream4_shard_smoke.py

echo "Step 8: Stream5 exchange smoke (WORLD_SIZE=1)..."
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python tests/stream5_exchange_smoke.py

echo "Step 9: Dispatcher skeleton smoke..."
python tests/dispatcher_skeleton_smoke.py

echo ""
echo "=== ALL TESTS COMPLETE ==="
