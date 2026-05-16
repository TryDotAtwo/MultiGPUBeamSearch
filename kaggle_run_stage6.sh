#!/bin/bash
# Helper script to run Stage6 smoke tests on Kaggle

cd /kaggle/working

# Check if repository is already cloned
if [ -d "CayleyBeam100H100" ]; then
    cd CayleyBeam100H100
    echo "Repository found. Pulling latest changes..."
    git pull origin master
else
    echo "Cloning repository from GitHub..."
    git clone https://github.com/trydotatwo/CayleyBeam100H100.git
    cd CayleyBeam100H100
fi

echo ""
echo "=========================================="
echo "Stage6 Dispatcher Skeleton Smoke Tests"
echo "=========================================="
echo ""

bash run_stage6_smoke_tests.sh
