#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/synccheck-stage
out="$root/partial-results"
mkdir -p "$out"
finish() {
    code=$?
    trap - EXIT
    printf 'EXIT_CODE=%s\n' "$code" >> "$out/status.txt"
    tar -czf "$root/partial-evidence.tar.gz" -C "$root" partial-results
    sha256sum "$root/partial-evidence.tar.gz" > "$root/partial-evidence.sha256"
    exit "$code"
}
trap finish EXIT
printf '%s  %s\n' "$1" "$root/partial-diagnostic.cu" | sha256sum -c - > "$out/source-check.txt"
cp "$root/partial-diagnostic.cu" "$out/"
cp "$root/run-partial.sh" "$out/"
echo BUILD_START >> "$out/status.txt"
nvcc -O3 -lineinfo -std=c++17 -arch=sm_86 -I"$root/source" \
    -DBEAM_STATE_LOGICAL_BYTES=96 -DBEAM_STATE_PHYSICAL_BYTES=112 -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=24 \
    "$root/partial-diagnostic.cu" -o "$root/partial-diagnostic" > "$out/build.log" 2>&1
sha256sum "$root/partial-diagnostic" > "$out/binary-sha256.txt"
for n in 1 1025; do
    for mode in plain conditional first; do
        for active in 1 65 256; do
            for tool in direct synccheck; do
                name="$mode-partial$active-$n-$tool"
                echo "START $name" >> "$out/status.txt"
                set +e
                if [[ "$tool" == direct ]]; then
                    timeout 100 "$root/partial-diagnostic" "$mode-partial$active" "$n" > "$out/$name.log" 2>&1
                else
                    timeout 100 compute-sanitizer --tool "$tool" --error-exitcode=86 \
                        "$root/partial-diagnostic" "$mode-partial$active" "$n" > "$out/$name.log" 2>&1
                fi
                code=$?
                set -e
                printf '%s\n' "$code" > "$out/$name.exit"
                echo "DONE $name exit=$code" >> "$out/status.txt"
            done
        done
    done
    name="first-partial1-$n-memcheck"
    echo "START $name" >> "$out/status.txt"
    set +e
    timeout 100 compute-sanitizer --tool memcheck --error-exitcode=86 \
        "$root/partial-diagnostic" first-partial1 "$n" > "$out/$name.log" 2>&1
    code=$?
    set -e
    printf '%s\n' "$code" > "$out/$name.exit"
    echo "DONE $name exit=$code" >> "$out/status.txt"
done
echo COMPLETE >> "$out/status.txt"
