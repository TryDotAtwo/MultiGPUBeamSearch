#!/bin/bash
set -Eeuo pipefail
root=/workspace/beam-results/native-generator
work=/workspace/beam-work
base=/workspace/beam-base
mkdir -p "$root/timing" "$root/timing-source"
cp "$work/tests/native_generator_parse_timing.cpp" "$work/tests/native_generator_parse_timing.py" "$root/timing-source/"
sha256sum "$root"/timing-source/* > "$root/timing-source/SHA256SUMS"
for shape in 96-112-24 150-160-255; do
    IFS=- read -r logical physical count <<< "$shape"
    defs=(-DBEAM_STATE_LOGICAL_BYTES="$logical" -DBEAM_STATE_PHYSICAL_BYTES="$physical" -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT="$count")
    g++ -std=c++20 -O2 "${defs[@]}" "$work/tests/native_generator_parse_timing.cpp" "$base/src/native_input.cpp" -o "$root/timing/old-$shape"
    g++ -std=c++20 -O2 "${defs[@]}" -DBEAM_BENCH_COMBINED "$work/tests/native_generator_parse_timing.cpp" "$work/src/native_input.cpp" -o "$root/timing/new-$shape"
    python "$work/tests/native_generator_parse_timing.py" --baseline "$root/timing/old-$shape" --candidate "$root/timing/new-$shape" --fixture "$root/cpu-$shape/moves.json" --out "$root/timing/$shape.json"
done
echo NATIVE_GENERATOR_UNINSTRUMENTED_TIMING_OK
