#!/bin/bash
set -Eeuo pipefail
root=/workspace/beam-results/native-generator
work=/workspace/beam-work
mkdir -p "$root/asan-tests"
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
    -DBEAM_STATE_LOGICAL_BYTES=120 -DBEAM_STATE_PHYSICAL_BYTES=128 \
    -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=18 \
    "$work/tests/native_input_tests.cpp" "$work/src/native_input.cpp" -o /workspace/tmp/native-input-asan
export ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1
(cd "$root/asan-tests" && /workspace/tmp/native-input-asan)
python "$work/tests/native_generator_snapshot_fifo.py" /workspace/tmp/native-input-asan \
    --state-len 120 --moves 18 --tmpdir /workspace/tmp
echo NATIVE_INPUT_ASAN_UBSAN_OK
