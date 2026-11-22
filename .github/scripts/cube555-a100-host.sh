#!/usr/bin/env bash
set -euo pipefail
archive=cube555-standalone-linux-x86_64-py311.tar
mkdir -p evidence python-tool
evidence="$PWD/evidence"

collect_evidence() {
    if [[ -d cube555-standalone/beam_run ]]; then
        cd cube555-standalone
        find beam_run -type f -size -4M \
            \( -name adapter-result.json -o -name '*capacity*.json' -o -name '*budget*.json' \
               -o -name runtime-config.json -o -name 'verification*.json' \
               -o -path '*/profile-grids/*.json' -o -name '*calibration*.json' \) -print0 \
            | tar -cf "$evidence/native-receipts.tar" --null -T -
    fi
}
trap 'status=$?; collect_evidence || true; exit "$status"' EXIT

sha256sum -c "$archive.sha256" | tee "$evidence/archive-checksums.txt"
tar -xf "$archive"
nvcc --version | tee "$evidence/nvcc.txt"
for tool in curl c++ make flock; do
    if ! command -v "$tool" >/dev/null; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get -qq update
        apt-get -qq install -y --no-install-recommends ca-certificates curl g++ make util-linux
        break
    fi
done
curl --fail --silent --show-error --location --retry 2 --max-time 90 \
    https://github.com/astral-sh/uv/releases/download/0.8.15/uv-x86_64-unknown-linux-musl.tar.gz \
    | tar -xz -C python-tool --strip-components=1
./python-tool/uv python install 3.11
export PYTHON="$(./python-tool/uv python find 3.11)"
"$PYTHON" --version | tee "$evidence/python.txt"
cd cube555-standalone
root="$PWD"
trap 'status=$?; cd "$root/.."; collect_evidence || true; exit "$status"' EXIT

bash run.sh --check |& tee "$evidence/check.log"
for stage in cold warm; do
    echo "Running $stage beam_width=max on $(date -u +%FT%TZ)"
    started="$(date +%s)"
    bash run.sh --beam-width max --max-steps 2 --bfs-radius 0 |& tee "$evidence/$stage.log"
    printf '%s\n' "$(( $(date +%s) - started ))" > "$evidence/$stage-seconds.txt"
    grids=(beam_run/cache/profile-grids/*.json)
    [[ "${#grids[@]}" == 1 ]]
    cp "${grids[0]}" "$evidence/$stage-grid.json"
done

"$PYTHON" - "$evidence" <<'PY'
import json
import sys
from pathlib import Path

evidence = Path(sys.argv[1])
cold = json.loads((evidence / "cold-grid.json").read_text())
warm = json.loads((evidence / "warm-grid.json").read_text())
assert cold["max"]["effective_beam"] == warm["max"]["effective_beam"]
assert cold["capacity"] == warm["capacity"]
assert cold["capacity"]["search_complete"] is True
assert cold["max"]["effective_beam"] <= cold["capacity"]["admitted_beam"]
assert cold["grid_reused"] is False and warm["grid_reused"] is True
assert set(cold["profiles"]) == set(warm["profiles"])
assert all(value["measurement_scope"] == "exact_admitted_component_capacities"
           for value in cold["profiles"].values())
assert warm["last_search"]["requested_beam"] == cold["max"]["effective_beam"]
assert warm["last_search"]["effective_beam"] == cold["max"]["effective_beam"]
assert warm["last_search"]["inference_cache_hit"] and warm["last_search"]["pipeline_cache_hit"]
print(f"Native memory admission: {cold['capacity']['admitted_beam']:,}; "
      f"rejected execution widths: {cold['capacity']['execution_rejected_beams']}")
for stage, grid in (("cold", cold), ("warm", warm)):
    print(f"{stage}: beam {grid['max']['effective_beam']:,}; "
          f"inference cache {grid['last_search']['inference_cache_hit']}; "
          f"pipeline cache {grid['last_search']['pipeline_cache_hit']}; "
          f"{len(grid['profiles'])} numeric profiles; "
          f"{(evidence / (stage + '-seconds.txt')).read_text().strip()} s elapsed")
PY
