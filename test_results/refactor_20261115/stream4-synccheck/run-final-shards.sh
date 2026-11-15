#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/synccheck-stage
out="$root/final-shards-results"
mkdir -p "$out"
finish() {
    code=$?
    trap - EXIT
    printf 'EXIT_CODE=%s\n' "$code" >> "$out/status.txt"
    tar -czf "$root/final-shards-evidence.tar.gz" -C "$root" final-shards-results
    sha256sum "$root/final-shards-evidence.tar.gz" > "$root/final-shards-evidence.sha256"
    exit "$code"
}
trap finish EXIT
printf '%s  %s\n' "$1" "$root/stream4_logical_union_cuda_tests.cu" | sha256sum -c - > "$out/logical-source-check.txt"
cp "$root/stream4_logical_union_cuda_tests.cu" "$root/candidate/tests/"
cp "$root/run-final-shards.sh" "$out/"
sha256sum "$root/candidate/tests/stream4_dirty_merge_cuda_tests.cu" \
    "$root/candidate/tests/stream4_logical_union_cuda_tests.cu" "$root/source/cuda/stream4.cu" > "$out/source-sha256.txt"
state_flags=(-DBEAM_STATE_LOGICAL_BYTES=96 -DBEAM_STATE_PHYSICAL_BYTES=112 -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=24)
for cap in 1 2 3 4 257; do
    echo "BUILD fixture $cap" >> "$out/status.txt"
    nvcc -O3 -lineinfo --expt-relaxed-constexpr -std=c++17 -arch=sm_86 \
        -I"$root/candidate/src" -I"$root/candidate/cuda" "${state_flags[@]}" -DBEAM_TEST_UNION_CAPACITY="$cap" \
        -c "$root/candidate/tests/stream4_dirty_merge_cuda_tests.cu" -o "$root/final-fixture-$cap.o" > "$out/fixture-$cap-build.log" 2>&1
    for variant in baseline-dirty baseline-index; do
        nvcc -arch=sm_86 "$root/final-fixture-$cap.o" "$root/$variant.o" -o "$root/final-$variant-$cap" > "$out/$variant-$cap-link.log" 2>&1
        sha256sum "$root/final-$variant-$cap" >> "$out/binary-sha256.txt"
    done
done
run_case() {
    local variant="$1" cap="$2" mode="$3" tool="$4" name="$1-$2-$3-$4"
    echo "START $name" >> "$out/status.txt"
    set +e
    if [[ "$tool" == direct ]]; then
        timeout 150 "$root/final-$variant-$cap" "--$mode-only" > "$out/$name.log" 2>&1
    else
        timeout 150 compute-sanitizer --tool "$tool" --error-exitcode=86 \
            "$root/final-$variant-$cap" "--$mode-only" > "$out/$name.log" 2>&1
    fi
    local code=$?
    set -e
    printf '%s\n' "$code" > "$out/$name.exit"
    echo "DONE $name exit=$code" >> "$out/status.txt"
}
for variant in baseline-dirty baseline-index; do
    for cap in 1 2 3 4 257; do
        for mode in unbounded bounded; do run_case "$variant" "$cap" "$mode" direct; done
    done
    for cap in 1 257; do run_case "$variant" "$cap" bounded memcheck; done
done
echo COMPLETE >> "$out/status.txt"
