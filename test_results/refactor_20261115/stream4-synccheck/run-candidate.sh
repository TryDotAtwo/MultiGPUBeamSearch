#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/synccheck-stage
out="$root/candidate-results"
mkdir -p "$out" "$root/candidate"
finish() {
    code=$?
    trap - EXIT
    printf 'EXIT_CODE=%s\n' "$code" >> "$out/status.txt"
    tar -czf "$root/candidate-evidence.tar.gz" -C "$root" candidate-results
    sha256sum "$root/candidate-evidence.tar.gz" > "$root/candidate-evidence.sha256"
    exit "$code"
}
trap finish EXIT
cd "$root"
printf '%s  %s\n' "$1" "$root/overlay.tar" | sha256sum -c - > "$out/overlay-check.txt"
cp -a "$root/source/." "$root/candidate/"
tar xf "$root/overlay.tar" -C "$root/candidate"
cp "$root/run-candidate.sh" "$out/"
sha256sum candidate/cuda/stream4.cu candidate/tests/stream4_dirty_merge_cuda_tests.cu > "$out/overlay-files-sha256.txt"
{ nvcc --version; compute-sanitizer --version; nvidia-smi --query-gpu=index,name,driver_version --format=csv; } > "$out/software.txt"
state_flags=(-DBEAM_STATE_LOGICAL_BYTES=96 -DBEAM_STATE_PHYSICAL_BYTES=112 -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=24)
common=(-O3 -lineinfo --expt-relaxed-constexpr -std=c++17 -arch=sm_86 -I"$root/candidate/src" -I"$root/candidate/cuda" "${state_flags[@]}")
for variant in baseline-dirty candidate-dirty candidate-index; do
    source_file="$root/candidate/cuda/stream4.cu"
    dirty=1; index=0
    if [[ "$variant" == baseline-dirty ]]; then source_file="$root/source/cuda/stream4.cu"; fi
    if [[ "$variant" == candidate-index ]]; then dirty=0; index=1; fi
    echo "BUILD $variant" >> "$out/status.txt"
    nvcc "${common[@]}" -DBEAM_EXPERIMENTAL_RADIX128=1 -DBEAM_EXPERIMENTAL_DIRTY_MERGE="$dirty" \
        -DBEAM_EXPERIMENTAL_INDEX_SORT="$index" -DBEAM_EXPERIMENTAL_STREAM4_SCAN=0 \
        -c "$source_file" -o "$root/$variant.o" > "$out/$variant-build.log" 2>&1
done
for cap in 1 2 3 4 257; do
    echo "BUILD fixture $cap" >> "$out/status.txt"
    nvcc "${common[@]}" -DBEAM_TEST_UNION_CAPACITY="$cap" \
        -c "$root/candidate/tests/stream4_dirty_merge_cuda_tests.cu" -o "$root/fixture-$cap.o" > "$out/fixture-$cap-build.log" 2>&1
    for variant in baseline-dirty candidate-dirty candidate-index; do
        nvcc -arch=sm_86 "$root/fixture-$cap.o" "$root/$variant.o" -o "$root/$variant-$cap" > "$out/$variant-$cap-link.log" 2>&1
        sha256sum "$root/$variant-$cap" >> "$out/binary-sha256.txt"
    done
done
run_case() {
    local variant="$1" cap="$2" mode="$3" tool="$4" name="$1-$2-$3-$4"
    echo "START $name" >> "$out/status.txt"
    set +e
    if [[ "$tool" == direct ]]; then
        timeout 150 "$root/$variant-$cap" "--$mode-only" > "$out/$name.log" 2>&1
    else
        timeout 150 compute-sanitizer --tool "$tool" --error-exitcode=86 \
            "$root/$variant-$cap" "--$mode-only" > "$out/$name.log" 2>&1
    fi
    local code=$?
    set -e
    printf '%s\n' "$code" > "$out/$name.exit"
    echo "DONE $name exit=$code" >> "$out/status.txt"
}
for variant in baseline-dirty candidate-dirty candidate-index; do
    for cap in 1 2 3 4 257; do
        for mode in unbounded bounded; do
            run_case "$variant" "$cap" "$mode" direct
        done
    done
done
for cap in 1 2 4; do run_case baseline-dirty "$cap" bounded synccheck; done
for variant in candidate-dirty candidate-index; do
    for cap in 1 2 3 4 257; do run_case "$variant" "$cap" bounded synccheck; done
    for cap in 1 4 257; do run_case "$variant" "$cap" bounded memcheck; done
    run_case "$variant" 1 bounded racecheck
    run_case "$variant" 1 unbounded synccheck
done
echo COMPLETE >> "$out/status.txt"
