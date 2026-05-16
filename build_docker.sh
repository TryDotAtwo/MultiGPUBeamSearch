#!/bin/bash
cd /workspace/CayleyBeam100H100
python setup.py build_ext --inplace
echo "BUILD_COMPLETE_$?"
