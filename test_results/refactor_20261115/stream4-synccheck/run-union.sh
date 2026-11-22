#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/synccheck-stage
out="$root/union-results"
mkdir -p "$out" "$root/union-source/tests"
finish() {
    code=$?
    trap - EXIT
    printf 'EXIT_CODE=%s\n' "$code" >> "$out/status.txt"
    tar -czf "$root/union-evidence.tar.gz" -C "$root" union-results
    sha256sum "$root/union-evidence.tar.gz" > "$root/union-evidence.sha256"
    exit "$code"
}
trap finish EXIT
printf '%s  %s\n' "$1" "$root/stream4_logical_union_cuda_tests.cu" | sha256sum -c - > "$out/test-source-check.txt"
cp "$root/stream4_logical_union_cuda_tests.cu" "$root/union-source/tests/"
cp "$root/stream4_logical_union_cuda_tests.cu" "$out/"
cp "$root/run-union.sh" "$out/"
ln -s ../candidate/cuda "$root/union-source/cuda"
state_flags=(-DBEAM_STATE_LOGICAL_BYTES=96 -DBEAM_STATE_PHYSICAL_BYTES=112 -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=24)
echo BUILD_START >> "$out/status.txt"
common=(-O3 -lineinfo --expt-relaxed-constexpr -std=c++17 -arch=sm_86
    -I"$root/candidate/src" -I"$root/candidate/cuda" "${state_flags[@]}")
nvcc "${common[@]}" -DBEAM_EXPERIMENTAL_RADIX128=1 -DBEAM_EXPERIMENTAL_DIRTY_MERGE=0 \
    -DBEAM_EXPERIMENTAL_INDEX_SORT=1 -DBEAM_EXPERIMENTAL_STREAM4_SCAN=0 \
    -c "$root/source/cuda/stream4.cu" -o "$root/baseline-index.o" > "$out/baseline-index-build.log" 2>&1
for variant in baseline-index candidate-index; do
nvcc -O3 -lineinfo --expt-relaxed-constexpr -std=c++17 -arch=sm_86 \
    -I"$root/candidate/src" -I"$root/candidate/cuda" "${state_flags[@]}" \
    -DBEAM_TEST_LOGICAL_UNION=1 -DBEAM_TEST_UNION_GRAPH=1 -DBEAM_TEST_UNION_SORT_BOUND=1 -DBEAM_TEST_INDEX_SORT=1 \
    "$root/union-source/tests/stream4_logical_union_cuda_tests.cu" "$root/$variant.o" \
    -o "$root/union-test-$variant" > "$out/$variant-build.log" 2>&1
sha256sum "$root/union-test-$variant" "$root/$variant.o" >> "$out/binary-sha256.txt"
for tool in direct memcheck synccheck; do
    echo "START $variant $tool" >> "$out/status.txt"
    set +e
    if [[ "$tool" == direct ]]; then
        timeout 180 "$root/union-test-$variant" > "$out/$variant-$tool.log" 2>&1
    else
        timeout 180 compute-sanitizer --tool "$tool" --error-exitcode=86 \
            "$root/union-test-$variant" > "$out/$variant-$tool.log" 2>&1
    fi
    code=$?
    set -e
    printf '%s\n' "$code" > "$out/$variant-$tool.exit"
    echo "DONE $variant $tool exit=$code" >> "$out/status.txt"
done
done
echo COMPLETE >> "$out/status.txt"
