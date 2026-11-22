#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/synccheck-stage
out="$root/prefix-results"
mkdir -p "$out"
finish() {
    code=$?
    trap - EXIT
    printf 'EXIT_CODE=%s\n' "$code" >> "$out/status.txt"
    tar -czf "$root/prefix-evidence.tar.gz" -C "$root" prefix-results
    sha256sum "$root/prefix-evidence.tar.gz" > "$root/prefix-evidence.sha256"
    exit "$code"
}
trap finish EXIT
printf '%s  %s\n' "$1" "$root/prefix-diagnostic.cu" | sha256sum -c - > "$out/source-check.txt"
cp "$root/prefix-diagnostic.cu" "$out/"
cp "$root/run-prefix.sh" "$out/"
echo BUILD_START >> "$out/status.txt"
nvcc -O3 -lineinfo -std=c++17 -arch=sm_86 -I"$root/source" \
    -DBEAM_STATE_LOGICAL_BYTES=96 -DBEAM_STATE_PHYSICAL_BYTES=112 -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=24 \
    "$root/prefix-diagnostic.cu" -o "$root/prefix-diagnostic" > "$out/build.log" 2>&1
sha256sum "$root/prefix-diagnostic" > "$out/binary-sha256.txt"
for n in 1 1025; do
    for mode in plain-prefix conditional-prefix first-prefix last-prefix; do
        for tool in direct memcheck synccheck; do
            name="$mode-$n-$tool"
            echo "START $name" >> "$out/status.txt"
            set +e
            if [[ "$tool" == direct ]]; then
                timeout 100 "$root/prefix-diagnostic" "$mode" "$n" > "$out/$name.log" 2>&1
            else
                timeout 100 compute-sanitizer --tool "$tool" --error-exitcode=86 \
                    "$root/prefix-diagnostic" "$mode" "$n" > "$out/$name.log" 2>&1
            fi
            code=$?
            set -e
            printf '%s\n' "$code" > "$out/$name.exit"
            echo "DONE $name exit=$code" >> "$out/status.txt"
        done
    done
done
echo COMPLETE >> "$out/status.txt"
