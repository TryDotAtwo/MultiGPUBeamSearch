#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/synccheck-stage
source_root="$root/source"
out="$root/results"
mkdir -p "$out"
finish() {
    code=$?
    trap - EXIT
    printf 'EXIT_CODE=%s\n' "$code" >> "$out/status.txt"
    tar -czf "$root/evidence.tar.gz" -C "$root" results
    sha256sum "$root/evidence.tar.gz" > "$root/evidence.sha256"
    exit "$code"
}
trap finish EXIT
cd "$source_root"
printf '%s  %s\n' "$1" "$root/source.tar" | sha256sum -c - > "$out/source-archive-check.txt"
printf '%s  %s\n' "$2" "$root/chained-diagnostic.cu" | sha256sum -c - > "$out/diagnostic-check.txt"
cp "$root/chained-diagnostic.cu" "$out/"
cp "$root/run.sh" "$out/"
sha256sum cuda/stream4.cu cuda/stream4.hpp cuda/radix128.cuh tests/stream4_dirty_merge_cuda_tests.cu tests/stream4_logical_union_cuda_tests.cu > "$out/key-source-sha256.txt"
{ nvcc --version; compute-sanitizer --version; nvidia-smi --query-gpu=index,name,driver_version --format=csv; } > "$out/software.txt"
state_flags=(-DBEAM_STATE_LOGICAL_BYTES=96 -DBEAM_STATE_PHYSICAL_BYTES=112 -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=24)
echo DIAGNOSTIC_BUILD_START >> "$out/status.txt"
nvcc -O3 -lineinfo -std=c++17 -arch=sm_86 -I"$source_root" "${state_flags[@]}" \
    "$root/chained-diagnostic.cu" -o "$root/chained-diagnostic" > "$out/diagnostic-build.log" 2>&1
sha256sum "$root/chained-diagnostic" > "$out/diagnostic-binary-sha256.txt"
echo DIAGNOSTIC_BUILD_OK >> "$out/status.txt"
for n in 1 1025; do
    for mode in plain conditional first last; do
        for tool in direct memcheck synccheck; do
            echo "START minimal $mode $n $tool" >> "$out/status.txt"
            set +e
            if [[ "$tool" == direct ]]; then
                timeout 100 "$root/chained-diagnostic" "$mode" "$n" > "$out/minimal-$mode-$n-$tool.log" 2>&1
            else
                timeout 100 compute-sanitizer --tool "$tool" --error-exitcode=86 \
                    "$root/chained-diagnostic" "$mode" "$n" > "$out/minimal-$mode-$n-$tool.log" 2>&1
            fi
            code=$?
            set -e
            printf '%s\n' "$code" > "$out/minimal-$mode-$n-$tool.exit"
            echo "DONE minimal $mode $n $tool exit=$code" >> "$out/status.txt"
        done
    done
done
echo FIXTURE_BUILD_START >> "$out/status.txt"
set +e
nvcc -O3 -lineinfo --expt-relaxed-constexpr -std=c++17 -arch=sm_86 -I"$source_root/src" -I"$source_root/cuda" \
    "${state_flags[@]}" -DBEAM_TEST_UNION_CAPACITY=1 -DBEAM_EXPERIMENTAL_RADIX128=1 \
    -DBEAM_EXPERIMENTAL_INDEX_SORT=0 -DBEAM_EXPERIMENTAL_DIRTY_MERGE=1 -DBEAM_EXPERIMENTAL_STREAM4_SCAN=0 \
    tests/stream4_dirty_merge_cuda_tests.cu cuda/stream4.cu -o "$root/stream4-cap1" > "$out/fixture-build.log" 2>&1
build_code=$?
set -e
printf '%s\n' "$build_code" > "$out/fixture-build.exit"
if [[ "$build_code" == 0 ]]; then
    sha256sum "$root/stream4-cap1" > "$out/fixture-binary-sha256.txt"
    for mode in unbounded bounded; do
        for tool in direct memcheck synccheck; do
            echo "START fixture $mode $tool" >> "$out/status.txt"
            set +e
            if [[ "$tool" == direct ]]; then
                timeout 120 "$root/stream4-cap1" "--$mode-only" > "$out/fixture-$mode-$tool.log" 2>&1
            else
                timeout 120 compute-sanitizer --tool "$tool" --error-exitcode=86 \
                    "$root/stream4-cap1" "--$mode-only" > "$out/fixture-$mode-$tool.log" 2>&1
            fi
            code=$?
            set -e
            printf '%s\n' "$code" > "$out/fixture-$mode-$tool.exit"
            echo "DONE fixture $mode $tool exit=$code" >> "$out/status.txt"
        done
    done
else
    echo "FIXTURE_BUILD_FAILED exit=$build_code" >> "$out/status.txt"
fi
echo COMPLETE >> "$out/status.txt"
